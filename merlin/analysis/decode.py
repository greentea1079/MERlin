import numpy as np
import pandas
import os
import tempfile
import cv2

from scipy.spatial import cKDTree
from skimage import measure

from merlin.core import dataset
from merlin.core import analysistask
from merlin.util import decoding
from merlin.util import barcodedb
from merlin.data.codebook import Codebook


class Decode(analysistask.ParallelAnalysisTask):

    """
    An analysis task that extracts barcodes from images.
    """

    def __init__(self, dataSet: dataset.ImageDataSet,
                 parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'crop_width' not in self.parameters:
            self.parameters['crop_width'] = 100
        if 'write_decoded_images' not in self.parameters:
            self.parameters['write_decoded_images'] = True
        if 'minimum_area' not in self.parameters:
            self.parameters['minimum_area'] = 0
        if 'distance_threshold' not in self.parameters:
            self.parameters['distance_threshold'] = 0.5167
        if 'lowpass_sigma' not in self.parameters:
            self.parameters['lowpass_sigma'] = 1
        if 'decode_3d' not in self.parameters:
            self.parameters['decode_3d'] = False
        if 'memory_map' not in self.parameters:
            self.parameters['memory_map'] = False

        self.cropWidth = self.parameters['crop_width']
        self.imageSize = dataSet.get_image_dimensions()

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        dependencies = [self.parameters['preprocess_task'],
                        self.parameters['optimize_task'],
                        self.parameters['global_align_task']]

        if 'segment_task' in self.parameters:
            dependencies += [self.parameters['segment_task']]

        return dependencies

    def get_codebook(self) -> Codebook:
        preprocessTask = self.dataSet.load_analysis_task(
            self.parameters['preprocess_task'])
        return preprocessTask.get_codebook()

    def _run_analysis(self, fragmentIndex):
        """This function decodes the barcodes in a fov and saves them to the
        barcode database.
        """
        preprocessTask = self.dataSet.load_analysis_task(
                self.parameters['preprocess_task'])
        optimizeTask = self.dataSet.load_analysis_task(
                self.parameters['optimize_task'])
        decode3d = self.parameters['decode_3d']

        lowPassSigma = self.parameters['lowpass_sigma']

        codebook = self.get_codebook()
        decoder = decoding.PixelBasedDecoder(codebook)
        scaleFactors = optimizeTask.get_scale_factors()
        backgrounds = optimizeTask.get_backgrounds()
        chromaticCorrector = optimizeTask.get_chromatic_corrector()

        zPositionCount = len(self.dataSet.get_z_positions_for_codebook(
            cb=codebook))
        bitCount = codebook.get_bit_count()
        imageShape = self.dataSet.get_image_dimensions()
        decodedImages = np.zeros((zPositionCount, *imageShape), dtype=np.int16)
        magnitudeImages = np.zeros((zPositionCount, *imageShape),
                                   dtype=np.float32)
        distances = np.zeros((zPositionCount, *imageShape), dtype=np.float32)

        if not decode3d:
            for zIndex in range(zPositionCount):
                di, pm, d = self._process_independent_z_slice(
                    fragmentIndex, zIndex, chromaticCorrector, scaleFactors,
                    backgrounds, preprocessTask, decoder
                )

                decodedImages[zIndex, :, :] = di
                magnitudeImages[zIndex, :, :] = pm
                distances[zIndex, :, :] = d

        else:
            with tempfile.TemporaryDirectory as tempDirectory:
                if self.parameters['memory_map']:
                    normalizedPixelTraces = np.memmap(
                        os.path.join(tempDirectory, 'pixel_traces.dat'),
                        mode='w+', dtype=np.float32,
                        shape=(zPositionCount, bitCount, *imageShape))
                else:
                    normalizedPixelTraces = np.zeros(
                        (zPositionCount, bitCount, *imageShape),
                        dtype=np.float32)

                for zIndex in range(zPositionCount):
                    imageSet = preprocessTask.get_processed_image_set(
                        fragmentIndex, zIndex, chromaticCorrector)
                    imageSet = imageSet.reshape(
                        (imageSet.shape[0], imageSet.shape[-2],
                         imageSet.shape[-1]))

                    di, pm, npt, d = decoder.decode_pixels(
                        imageSet, scaleFactors, backgrounds,
                        lowPassSigma=lowPassSigma,
                        distanceThreshold=self.parameters['distance_threshold'])

                    normalizedPixelTraces[zIndex, :, :, :] = npt
                    decodedImages[zIndex, :, :] = di
                    magnitudeImages[zIndex, :, :] = pm
                    distances[zIndex, :, :] = d

                self._extract_and_save_barcodes(
                    decoder, decodedImages, magnitudeImages,
                    normalizedPixelTraces,
                    distances, fragmentIndex)

                del normalizedPixelTraces

        if self.parameters['write_decoded_images']:
            self._save_decoded_images(
                fragmentIndex, zPositionCount, decodedImages, magnitudeImages,
                distances)

    def _process_independent_z_slice(
            self, fov: int, zIndex: int, chromaticCorrector, scaleFactors,
            backgrounds, preprocessTask, decoder):

        imageSet = preprocessTask.get_processed_image_set(
            fov, zIndex, chromaticCorrector)
        imageSet = imageSet.reshape(
            (imageSet.shape[0], imageSet.shape[-2], imageSet.shape[-1]))

        di, pm, npt, d = decoder.decode_pixels(
            imageSet, scaleFactors, backgrounds,
            lowPassSigma=self.parameters['lowpass_sigma'],
            distanceThreshold=self.parameters['distance_threshold'])
        self._extract_and_save_barcodes(
            decoder, di, pm, npt, d, fov, zIndex)

        return di, pm, d

    def get_barcode_database(self) -> barcodedb.BarcodeDB:
        return barcodedb.PyTablesBarcodeDB(self.dataSet, self)

    def _save_decoded_images(self, fov: int, zPositionCount: int,
                             decodedImages: np.ndarray,
                             magnitudeImages: np.ndarray,
                             distanceImages: np.ndarray) -> None:
        imageDescription = self.dataSet.analysis_tiff_description(
            zPositionCount, 3)
        with self.dataSet.writer_for_analysis_images(
                self, 'decoded', fov) as outputTif:
            for i in range(zPositionCount):
                outputTif.save(decodedImages[i].astype(np.float32),
                               photometric='MINISBLACK',
                               metadata=imageDescription)
                outputTif.save(magnitudeImages[i].astype(np.float32),
                               photometric='MINISBLACK',
                               metadata=imageDescription)
                outputTif.save(distanceImages[i].astype(np.float32),
                               photometric='MINISBLACK',
                               metadata=imageDescription)

    def _extract_and_save_barcodes(
            self, decoder: decoding.PixelBasedDecoder, decodedImage: np.ndarray,
            pixelMagnitudes: np.ndarray, pixelTraces: np.ndarray,
            distances: np.ndarray, fov: int, zIndex: int = None) -> None:

        globalTask = self.dataSet.load_analysis_task(
            self.parameters['global_align_task'])

        segmentTask = None
        if 'segment_task' in self.parameters:
            segmentTask = self.dataSet.load_analysis_task(
                self.parameters['segment_task'])

        minimumArea = self.parameters['minimum_area']

        self.get_barcode_database().write_barcodes(
            pandas.concat([decoder.extract_barcodes_with_index(
                i, decodedImage, pixelMagnitudes, pixelTraces, distances, fov,
                self.cropWidth, zIndex, globalTask, segmentTask, minimumArea)
                for i in range(self.get_codebook().get_barcode_count())]),
            fov=fov)


class HardDecode(analysistask.ParallelAnalysisTask):
    """
    An analysis task that extracts barcodes from images.
    Decoding is done in a "hard" manner wherein bits are called
    as on or off, selecting the number of bits with the highest signal
    based on the number of "on" bits in the codewords. This is intended
    as a fast algorithm of decoding with similar performance to the soft
    decoding algorithm.
    """

    def __init__(self, dataSet: dataset.ImageDataSet,
                 parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'crop_width' not in self.parameters:
            self.parameters['crop_width'] = 100
        if 'write_decoded_images' not in self.parameters:
            self.parameters['write_decoded_images'] = True
        if 'minimum_area' not in self.parameters:
            self.parameters['minimum_area'] = 0
        if 'lowpass_sigma' not in self.parameters:
            self.parameters['lowpass_sigma'] = 1

        self.cropWidth = self.parameters['crop_width']
        self.imageSize = dataSet.get_image_dimensions()

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        dependencies = [self.parameters['preprocess_task'],
                        self.parameters['optimize_task'],
                        self.parameters['global_align_task']]
        return dependencies

    def get_codebook(self) -> Codebook:
        preprocessTask = self.dataSet.load_analysis_task(
            self.parameters['preprocess_task'])
        return preprocessTask.get_codebook()

    def _run_analysis(self, fragmentIndex) -> None:
        """This function decodes the barcodes in a fov and saves them to the
        barcode database.
        """
        preprocessTask = self.dataSet.load_analysis_task(
            self.parameters['preprocess_task'])
        optimizeTask = self.dataSet.load_analysis_task(
            self.parameters['optimize_task'])

        scaleFactors = optimizeTask.get_scale_factors()
        chromaticCorrector = optimizeTask.get_chromatic_corrector()

        zPositionCount = len(self.dataSet.get_z_positions_for_codebook(
            cb=self.get_codebook()))
        bcs = self.get_codebook().get_barcodes()
        tree = cKDTree(data=bcs)
        imageDims = self.dataSet.get_image_dimensions()

        for zIndex in range(zPositionCount):
            cw, idx, pixelMagnitudes, scaledPixelTraces =\
                self._process_independent_z_slice(fragmentIndex, zIndex,
                                                  chromaticCorrector,
                                                  scaleFactors, preprocessTask)

            [d, i] = tree.query(cw, k=1)
            i = i + 1

            masked = np.where(d == 0, i, 0)
            rebuilt = masked[idx]
            rebuiltReshaped = np.reshape(rebuilt, tuple(imageDims))
            labels = measure.label(rebuiltReshaped, connectivity=2)

            rProps = measure.regionprops(labels, rebuiltReshaped)
            rProps2 = measure.regionprops(labels, np.reshape(pixelMagnitudes,
                                                             tuple(imageDims)))
            properties = self._parse_regionprops_info(rProps, rProps2, zIndex,
                                                      fragmentIndex)
            normalizedPixelTraces = scaledPixelTraces / pixelMagnitudes[:, None]

            meanDist, intensities = self._calculate_mean_distances(
                normalizedPixelTraces, labels, properties['barcode_id'].values)
            properties['mean_distance'] = meanDist.values.tolist()
            intensities.columns = ['intensity_{}'.format(x) for x
                                   in intensities.columns]
            intensities.index = properties.index
            tableOut = pandas.concat([properties,intensities],1)

            self.get_barcode_database().write_barcodes(tableOut,
                                                       fov=fragmentIndex)

    def _parse_regionprops_info(self, rProps, intensityrProps,
                                zIndex, fov) -> pandas.DataFrame:
        inputCentroids = np.array([[zIndex] + list(x.weighted_centroid)
                                   for x in rProps])
        inputCentroids[:, [0, 1, 2]] = inputCentroids[:, [0, 2, 1]]

        globalAlignmentTask = self.dataSet.load_analysis_task(
            self.parameters['global_align_task'])

        globalCentroids = np.array(
            [globalAlignmentTask.fov_coordinates_to_global(fov, tuple(x))
             for x in inputCentroids])

        bcIndex = [x.max_intensity - 1 for x in rProps]
        mean_intensity = [x.mean_intensity for x in intensityrProps]
        max_intensity = [x.max_intensity for x in intensityrProps]
        area = [x.area for x in rProps]
        min_distance = [-1]*len(rProps)
        x = inputCentroids[:, 1]
        y = inputCentroids[:, 2]
        z = inputCentroids[:, 0]
        global_x = globalCentroids[:, 1]
        global_y = globalCentroids[:, 2]
        global_z = globalCentroids[:, 0]
        cell_index = [0]*len(rProps)

        return pandas.DataFrame(data=[bcIndex, [fov]*len(rProps),
                                      mean_intensity, max_intensity, area,
                                      min_distance, x, y, z, global_x, global_y,
                                      global_z, cell_index],
                            index=['barcode_id', 'fov', 'mean_intensity',
                                   'max_intensity', 'area', 'min_distance', 'x',
                                   'y', 'z', 'global_x', 'global_y', 'global_z',
                                   'cell_index']).T

    def _process_independent_z_slice(self, fov: int, zIndex: int,
                                     chromaticCorrector, scaleFactors,
                                     preprocessTask):

        imageSet = preprocessTask.get_processed_image_set(
            fov, zIndex, chromaticCorrector)
        imageSet = imageSet.reshape(
            (imageSet.shape[0], imageSet.shape[-2], imageSet.shape[-1]))
        filteredImages = np.zeros(imageSet.shape, dtype=np.float32)

        filterSize = int(2 * np.ceil(2 * int(self.parameters['lowpass_sigma']))
                         + 1)

        for i in range(imageSet.shape[0]):
            filteredImages[i, :, :] = cv2.GaussianBlur(
                imageSet[i, :, :], (filterSize, filterSize), 1)

        pixelTraces = np.reshape(filteredImages,
                                 (filteredImages.shape[0],
                                  np.prod(filteredImages.shape[1:])))
        scaledPixelTraces = np.transpose(np.array([(p) / s for p, s in zip(
            pixelTraces, scaleFactors)]))
        pixelMagnitudes = np.array([np.linalg.norm(x)
                                    for x in scaledPixelTraces],
                                   dtype=np.float32)
        pixelMagnitudes[pixelMagnitudes == 0] = 1

        cb = self.get_codebook()
        onBitCount = cb.get_on_bit_count()
        allBitCount = cb.get_bit_count()

        exactCodewords = [np.argsort(x)[-4:] for x in scaledPixelTraces]
        cw, idx = np.unique(exactCodewords, return_inverse=True, axis=0)
        cw = [[1 if i in x else 0 for i in range(allBitCount)] for x in cw]

        return cw, idx, pixelMagnitudes, scaledPixelTraces

    def get_barcode_database(self) -> barcodedb.BarcodeDB:
        return barcodedb.PyTablesBarcodeDB(self.dataSet, self)

    def _calculate_mean_distances(self, normPixelTraces, labels, codewords):
        normPixelTraces = pandas.DataFrame(normPixelTraces)
        normPixelTraces['labels'] = np.reshape(labels, normPixelTraces.shape[0])
        means = normPixelTraces.groupby('labels').mean().iloc[1:, :]

        decodingMatrix = self.get_codebook().\
            get_weighted_barcode_decoding_matrix()

        cwHolding = pandas.DataFrame(decodingMatrix[np.array(codewords,
                                                             dtype = int)],
                                     index=means.index)
        diff = means - cwHolding
        diff = diff.apply(lambda x: x ** 2).sum(1).apply(np.sqrt)
        return diff, means
