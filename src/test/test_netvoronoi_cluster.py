# coding=gbk 
"""networkvoronoi unittest"""
import unittest
import netvoronoi_cluster as nv
import random
random.seed(10)
"""
several pre-processing steps are omitted here
1. Graph should be well connected
2. the graph shp and the event shp should be in the same projection system
"""
class NetworkVoronoi_Tester(unittest.TestCase):
    
    def setUp(self):
        print 'set up'
        
#    def test_crime_density(self):
##        G_weighted, G_origin = nv.net_density('data/pysal/streets.shp', 'data/pysal/crimes.shp', 200, 500, edge_weight_mode = "additive", normalize = [-100.0,100.0])
#        G_weighted, G_origin = nv.net_density('data/pysal/streets.shp', 'data/pysal/crimes.shp', 20, 50, normalize_method = "linear")
#        nv.netvoronoi(G_weighted, G_origin, 'data/pysal/stations.shp', 'data/pysal/cluster/netvoronoi_lincs_linear_no_station_wgt.shp','ID_test', None, None)
    
    def test_crime_lincs(self):
        G_weighted, G_origin = nv.net_lincs('../data/pysal/streets.shp', 'data/pysal/crimes.shp', 10, edge_weight_mode = "additive", normalize = [-100.0,100.0])
        nv.netvoronoi(G_weighted, G_origin, '../data/pysal/stations.shp', 'data/pysal/netvoronoi_lincs_linear_no_station_wgt.shp','ID_test', None, None)
   
suite = unittest.TestSuite()
test_classes = [NetworkVoronoi_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)