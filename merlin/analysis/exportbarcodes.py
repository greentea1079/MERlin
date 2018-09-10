from merlin.core import analysistask

class ExportBarcodes(analysistask.AnalysisTask):

    '''An analysis task that filters barcodes based on area and mean 
    intensity.
    '''

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.columns = self.parameters.get('columns', 
                ['barcode_id', 'global_x', 'global_y', 'cell_index'])
        self.excludeBlanks = self.parameters.get('exclude_blanks', True)

    def get_estimated_memory(self):
        return 5000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['filter_task']]

    def run_analysis(self):
        self.filterTask = self.dataSet.load_analysis_task(
                self.parameters['filter_task'])        

        barcodeData = self.filterTask.get_barcode_database() \
                .get_barcodes(columnList = self.columns)

        if self.excludeBlanks:
            codebook = self.dataSet.get_codebook()
            codingIDs = codebook[~codebook['name'].str.contains('Blank')]
            barcodeData = barcodeData[\
                    barcodeData['barcode_id'].isin(codingIDs.index)]

        self.dataSet.save_dataframe_to_csv(barcodeData, 'barcodes', self)
