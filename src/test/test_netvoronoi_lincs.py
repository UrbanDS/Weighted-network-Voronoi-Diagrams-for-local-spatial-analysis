"""networkvoronoi unittest"""
import unittest
import sys
sys.path.append('../')
from src import netvoronoi_lincs as nv
import random
random.seed(10)
"""
several pre-processing steps are omitted here
1. Graph should be well connected
2. the graph shp and the event shp should be in the same projection system
"""
class NetworkVoronoi_Tester(unittest.TestCase):
    
    def setUp(self):
        print('set up')
        
    def test_network_vornoi_pysal(self):
        nv.netvoronoi('../../data/pysal/streets.shp', '../../data/pysal/stations.shp', '../../data/pysal/netvoronoi_lincs_linear_no_station_wgt.shp','ID_test', None, None,'../../data/pysal/crimes.shp',normalize_method = "linear")
        nv.netvoronoi('../../data/pysal/streets.shp', '../../data/pysal/stations.shp', '../../data/pysal/netvoronoi_lincs_quantile_no_station_wgt.shp','ID_test', None, None,'../../data/pysal/crimes.shp')
        nv.netvoronoi('../../data/pysal/streets.shp', '../../data/pysal/stations.shp', '../../data/pysal/netvoronoi_lincs_quantile_additive_no_station_wgt.shp','ID_test', None, None,'../../data/pysal/crimes.shp',edge_weight_mode = "additive", normalize = [-100.0,100.0])
        nv.netvoronoi('../../data/pysal/streets.shp', '../../data/pysal/stations.shp', '../../data/pysal/netvoronoi_no_station_wgt.shp','ID_test', None, None)
        
#        nv.netvoronoi('../../data/pysal/streets.shp', '../../data/pysal/stations.shp', '../../data/pysal/netvoronoi_lincs_add_wgt.shp','ID_test', 'add_wgt', None,'../../data/pysal/crimes.shp')
#        nv.netvoronoi('../../data/pysal/streets.shp', '../../data/pysal/stations.shp', '../../data/pysal/netvoronoi_add_wgt.shp','ID_test', 'add_wgt', None)
#        
#        nv.netvoronoi('../../data/pysal/streets.shp', '../../data/pysal/stations.shp', '../../data/pysal/netvoronoi_lincs_mul_wgt.shp','ID_test', None, 'mul_wgt','../../data/pysal/crimes.shp')
#        nv.netvoronoi('../../data/pysal/streets.shp', '../../data/pysal/stations.shp', '../../data/pysal/netvoronoi_mul_wgt.shp','ID_test', None, 'mul_wgt')
#        
#        nv.netvoronoi('../../data/pysal/streets.shp', '../../data/pysal/stations.shp', '../../data/pysal/netvoronoi_lincs_all_wgts.shp','ID_test', 'add_wgt', 'mul_wgt','../../data/pysal/crimes.shp')
#        nv.netvoronoi('../../data/pysal/streets.shp', '../../data/pysal/stations.shp', '../../data/pysal/netvoronoi_all_wgts.shp','ID_test', 'add_wgt', 'mul_wgt')
    
        
suite = unittest.TestSuite()
test_classes = [NetworkVoronoi_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)