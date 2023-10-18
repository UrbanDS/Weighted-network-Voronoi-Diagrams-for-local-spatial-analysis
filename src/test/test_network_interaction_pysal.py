# coding=gbk 
"""networkvoronoi unittest"""
import unittest
import network_interaction as ni
#from pysal.spatial_dynamics import interaction
import network as pynet
import random
random.seed(10)
"""
several pre-processing steps are omitted here
1. Graph should be well connected
2. the graph shp and the event shp should be in the same projection system
"""
class NetworkVoronoi_Tester(unittest.TestCase):
    def setUp(self):
        self.events = ni.SpaceTimeEvents('../data/pysal/crimes', 'T')
        self.G = pynet.read_network('../data/pysal/streets.shp')

#    def test_knox(self):
#        result = ni.net_knox(self.events, 20, 5, self.G,'network', 99)
#        print result
        
#    def test_mantel(self):
#        result = ni.net_mantel(self.G,self.events, 99, scon=0.0,
#                                    spow=1.0, tcon=0.0, tpow=1.0)
#        print result
#        
#    def test_modified_knox(self):
#        result = ni.net_modified_knox(self.G,
#            self.events, delta=20, tau=5, permutations=99)
#        print result
#        
    def test_jacquez(self):
#        result = ni.net_jacquez(self.events, 8,self.G, 'network', 99)
        result = ni.net_jacquez(self.events, 8,None, 'manhatten', 99)
        print result
   
suite = unittest.TestSuite()
test_classes = [NetworkVoronoi_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)