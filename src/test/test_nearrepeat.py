'''
Created on 2013-6-3

'''
import network_interaction as ni
events = ni.SpaceTimeEvents('../data/pysal/crimes', 'T')
knox_result = ni.net_knox_grid(events, 300, 1, 100, 1, None, 'manhattan', 999)
#print knox_result['exps']
#print knox_result['pvalues']
#print knox_result['stats']
#knox_result = ni.net_knox(events, 300, 100, None, 'manhattan', 99)
print knox_result