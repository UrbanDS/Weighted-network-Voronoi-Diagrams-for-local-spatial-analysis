# coding=gbk 
"""networkvoronoi unittest"""
import unittest
import netvoronoi as nv
import random
random.seed(10)
"""
several pre-processing steps are omitted here
1. Graph should be well connected
2. the graph shp and the event shp should be in the same projection system
"""
class NetworkVoronoi_Tester(unittest.TestCase):
    
    def setUp(self):
        self.net = 'test/streets.shp'
        self.events = 'test/crimes.shp'
        
    def test_network_vornoi(self):
        nv.netvoronoi(self.net, self.events, "test/netvoronoi_no_wgt.shp","ID_test")
        
    def test_network_vornoi_add_wgt(self):
        #test with additive weight
        nv.netvoronoi(self.net, self.events, 'test/netvoronoi_add_wgt.shp', 'ID_test','add_wgt')
        
    def test_network_vornoi_mul_wgt(self):
        #test with multiplicative weight
        nv.netvoronoi(self.net, self.events, 'test/netvoronoi_mul_wgt.shp', 'ID_test',None,'mul_wgt')
        
    def test_network_vornoi_all_wgts(self):
        #test with both weights
        nv.netvoronoi(self.net, self.events, 'test/netvoronoi_all_wgts.shp', 'ID_test','add_wgt','mul_wgt')
        
suite = unittest.TestSuite()
test_classes = [NetworkVoronoi_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)