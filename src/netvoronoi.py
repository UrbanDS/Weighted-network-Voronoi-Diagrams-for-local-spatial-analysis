"""
A library for computing Voronoi diagram for network-constrained data.
Based on algorithms described in Spatial Analysis along Networks, 2012, Chapter 4

The library includes the additive weight and the multiplicative weight 
for the point events. It currently deal with only undirected network. 

The library relies on PySAL 1.4 (or higher), and network.py in the 
unreleased PySAL development version

"""
import network as pynet
import pysal.cg
from pysal.cg.shapes import Point, Chain
from datetime import datetime

class Event(Point):
    """
    The event (or point generator) that has its id and two weight fields (additive and multiplicative).

    Attributes
    ----------

    id              : Object
                      The identifier
    add_wgt         : float
                      The additive weight
    mul_wgt         : float
                      The multiplicative weight.
    """
    def __init__(self, loc, id, add_wgt = 0.0, mul_wgt = 1.0):
        self.id = id
        self.add_wgt = add_wgt
        self.mul_wgt = mul_wgt
        super(Event,self).__init__(loc)
    
    def __eq__(self, other):
        try:
            return (self.id) == (other.id)
        except AttributeError:
            return False
    
    def __str__(self):
        return str(self[:])+" id: "+str(self.id) + " additive weight: " + str(self.add_wgt)+" multiplicative weight: "+ str(self.mul_wgt)
   
def netvoronoi(network_file,event_file,output_file, id_field=None, add_w_field=None,mul_w_field=None):
    """
    Compute the network Voronoi diagram and write the result to a shp file.

    Parameters
    ----------
    
    network_file     : string
                       File path of the input network, must be projected
    event_file       : string  
                       File path to the input event points, must be in the same
                       projection system with network
    output_file      : string
                       File path to the output file
    id_field         : string
                       the identifier field for the events, created automatically 
                       if None is passed in
    add_w_field      : string
                       the additive weight field for the events. If set to None, 
                       the additive weight will be 0 for all events 
    mul_w_field      : string
                       the multiplicative weight field for the events. If set to 
                       None, the multiplicative weight will be 1 for all events
                           
    Examples
    --------
    
    No weights included
    
    >>> import netvoronoi as nv
    >>> nv.netvoronoi('streets.shp', 'crimes.shp', 'netvoronoi_no_wgt.shp','ID_test')
    Reading network streets.shp: 0.016 seconds
    Reading events crimes.shp: 0.031 seconds
    Snapping events to the network: 3.485 seconds
    Finding the nearest event for every node: 1.484 seconds
    Finding the nearest event for every edge (or new ones): 0.0 seconds
    Writing the result out netvoronoi_no_wgt.shp: 0.109 seconds
    Done
    
    Additive weight
    >>> import netvoronoi as nv
    >>> nv.netvoronoi('streets.shp', 'crimes.shp', 'netvoronoi_add_wgt.shp', 'ID_test','add_wgt')
    ...
    """
    
    st_time = datetime.now()
    G = pynet.read_network(network_file)
    print_info('Reading network '+network_file,st_time)
    
    st_time = datetime.now()
    # read the events in
    events, id_spec = read_events(event_file,id_field,add_w_field,mul_w_field)
    print_info('Reading events '+event_file,st_time)
    
    # get a mapping for the event points and the projected ones
    st_time = datetime.now() 
    evt_snapped_dic = snap_and_add(events, G)
    print_info('Snapping events to the network',st_time)
    
    # For each node in the graph, find the nearest event point 
    st_time = datetime.now() 
    distance_dic = {} #the nearest distance recorded for each node
    vertex_dic = {} #the node cover vertex_dic
    for e in events:
        distances = pynet.dijkstras(G, evt_snapped_dic[e.id])
        for k,v in distances.items():
            v = v * e.mul_wgt + e.add_wgt
            if distance_dic.get(k, 1e10) > v:
                distance_dic[k] = v
                vertex_dic[k] = e
    print_info('Finding the nearest event for every node',st_time)
    
    """
    traverse through all the edges, if the events attached by two end nodes are 
    the same, we say this edge belongs to that event. If not, find the break point 
    and form two new sub-edges.
    """
    st_time = datetime.now() 
    edge_belong_dic = {}
    used = {}
    for k, neighbors in G.items():
        assert not isinstance(k,Event)
        assert k in vertex_dic
             
        for n in neighbors:
            assert not isinstance(n,Event)
            assert n in vertex_dic
        
            """
            if the k,n pair is already processed, continue directly, this worked 
            since each pair appeared exactly twice
            """
            if k+n in used:
                continue
            
            used[n+k] = True
            
            if vertex_dic[k] == vertex_dic[n]:
                edge_belong_dic[(k,n)] = vertex_dic[k]
            else:
                # find the break point and form two new edges
                break_point = find_break_point(k, n, G[k][n], distance_dic[k], distance_dic[n], vertex_dic[k].mul_wgt, vertex_dic[n].mul_wgt)
                edge_belong_dic[(k,break_point)] = vertex_dic[k]
                edge_belong_dic[(break_point,n)] = vertex_dic[n]
    print_info('Finding the nearest event for every edge (or new ones)',st_time)
    
    #write the result to the shp file
    st_time = datetime.now() 
    write_edges_to_shp(edge_belong_dic, output_file, id_spec)
    print_info('Writing the result out '+output_file,st_time)
    print_info('Done')
   
def read_events(event_file, id_field=None, add_w_field=None,mul_w_field=None):
    """
    Read the event points from a shp file
    """
    events = []
    s = pysal.open(event_file)
    dbf = pysal.open(event_file[:-3] + 'dbf')
    if s.type != pysal.cg.shapes.Point:
        raise ValueError, 'File is not of type Point'
    
    fields = (id_field, add_w_field, mul_w_field)
    idxs = []
    for f in fields:
        if f == None:
            idxs.append(-1)
        else:
            try:
                idxs.append(dbf.header.index(f))
            except ValueError:
                raise Exception('filed '+f+' not found in '+event_file +' !')
    
    def field_val(r,idx,default):
        if idx>=0:
            return r[idx]
        else:
            return default

    i = 0
    for g, r in zip(s,dbf):
        events.append(Event(g,field_val(r,idxs[0],i),field_val(r,idxs[1],0.0),field_val(r,idxs[2],1.0)))
        i += 1
    
    id_spec = field_val(dbf.field_spec,idxs[0],('N', 9, 0))
    return events, id_spec


def snap_and_add(events, G):
    """
    Snap the event points to the nearest segments in the network. The projected 
    point will break the original edge into two sub-edges. Reordering of events 
    takes place if multiple event points are projected to the same segment. 
    
    The algorithm is based on the snapping routine in network.py(from PySAL)
    """
    evt_snapped_dic = {}
    
    snapper = pynet.Snapper(G)
    edge_dic = {}
    for e in events:
        snapped = snapper.snap(e)
        # get a order-independent key of the two end nodes (which form an edge)
        key = snapped[0] + snapped[1]
        if snapped[0]>snapped[1]:
            key = snapped[1] + snapped[0]
            snapped = (snapped[1],snapped[0],snapped[3],snapped[2]) # the last two number are actually not used
   
        edge_dic.setdefault(key,[])
        point = find_point_on_edge(snapped[0],snapped[1],snapped[2],snapped[2]+snapped[3])
        
        edge_dic[key].append(point)
        evt_snapped_dic[e.id] = point
        
    for key,events in edge_dic.items():
        st = key[0:2]
        end = key[2:]
        
        """
        Sort the events based on the distance to start node.
        The sort order is decided by the two end nodes.
        """
        events.sort(reverse = st > end)
        
        # remove the original edge
        del G[st][end]
        del G[end][st]
        
        size = len(events)
        for i in range(size):
            cur_e = events[i]
            
            G.setdefault(cur_e, {})
            
            # add the edge with previous node
            prev_point = st
            if i <> 0:
                prev_point = events[i-1]
            
            dis_st = pysal.cg.get_points_dist(Point(prev_point), Point(cur_e)) 
            G[cur_e][prev_point] = dis_st
            G[prev_point][cur_e] = dis_st
            
            # if it's the last one, add the edge connecting the end node
            if i == size-1:
                aft_point = end
                dis_end = pysal.cg.get_points_dist(Point(cur_e), Point(aft_point)) 
                G[cur_e][aft_point] = dis_end
                G[aft_point][cur_e] = dis_end
                
    return evt_snapped_dic

def find_break_point(p1, p2, edge_len, dp1, dp2,mul_e1=1.0,mul_e2=1.0):
    """
    Find a point on edge for two events with multiplicative weights (note that the 
    additive weight is already covered in dp1 and dp2)
    """
    #distance of the breakpoint from p1
    dis = (dp2 - dp1 + edge_len * mul_e2) / (mul_e1 + mul_e2)
    assert dis >= 0.0
    # calculate the absolute location
    return find_point_on_edge(p1,p2,dis,edge_len)

def find_point_on_edge(p1, p2, dis_p1,edge_len):
    """
    Find a point on edge based on the distance to the starting point
    """
    dis_propotion = dis_p1/edge_len
    return tuple( (b-a)*dis_propotion + a for a,b in zip(p1,p2) )

def write_edges_to_shp(edge_belong_dic, output_file, event_id_spec):
    """
    write the network Voronoi computation result to a SHP file 
    """
    if not output_file.lower().endswith('shp'):
        print 'filename would end with shp'
        return
    
    shp = pysal.open(output_file, 'w')
    dbf = pysal.open(output_file[:-3] + 'dbf', 'w')
    dbf.header = ['ID','EVENT']
    dbf.field_spec = [('N', 9, 0), event_id_spec]
    
    # traverse all the edges, write in the event id and the edge geometry.
    counter = 0
    for pair,event in edge_belong_dic.items():
        shp.write(Chain([Point(pair[0]), Point(pair[1])]))                    
        dbf.write([counter, event.id])
        counter += 1
        
    shp.close()
    dbf.close()

def print_info(info,cur_time = None):
    if cur_time <>None:
        delta = datetime.now() - cur_time
        combined = delta.seconds + delta.microseconds/1E6
        print info +": "+str(combined) + " seconds"
    else:
        print info