"""
A library for computing Voronoi diagram for network-constrained data.
The algorithm is originally described in Spatial Analysis along Networks, 2012, Chapter 4
In this implementation, clustering levels of network edges are incorporated. The edge weigth
can be multiplicative or additive.

The library includes the additive weight and the multiplicative weight 
for the point events. It currently deal with only undirected network. 

The library relies on PySAL 1.4 (or higher), and network.py and lincs.py in the 
unreleased PySAL development version

"""
import network as pynet
import libpysal.cg
from libpysal.cg.shapes import Point, Chain
from datetime import datetime
import lincs
import copy
import operator

class Station(Point):
    """
    The Station (or point generator) that has its id and two weight fields (additive and multiplicative).

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
        super(Station,self).__init__(loc)
    
    def __eq__(self, other):
        try:
            return (self.id) == (other.id)
        except AttributeError:
            return False
    
    def __str__(self):
        return str(self[:])+" id: "+str(self.id) + " additive weight: " + str(self.add_wgt)+" multiplicative weight: "+ str(self.mul_wgt)
   
def netvoronoi(network_file, station_file, output_file,
               id_field=None, add_w_field=None,mul_w_field=None, 
               event_file = None,weight = 'Distance-based',  distance_threshold = 500, 
               lisa_func = 'moran', sim_method = 'permutations', sim_num = 99, alpha = 0.05, edge_weight_mode = "multiplicative", normalize_method = "quantile", normalize = [0.5, 1.5]):
    """
    Compute the network Voronoi diagram and write the result to a shp file.

    Parameters
    ----------
    
    network_file     : string
                       File path of the input network, must be projected
    station_file     : string  
                       File path to the input station points, must be in the same
                       projection system with network
    output_file      : string
                       File path to the output file
    id_field         : string
                       the identifier field for the stations, created automatically 
                       if None is passed in
    add_w_field      : string
                       the additive weight field for the stations. If set to None, 
                       the additive weight will be 0 for all stations 
    mul_w_field      : string
                       the multiplicative weight field for the stations. If set to 
                       None, the multiplicative weight will be 1 for all stations
    event_file       : string  
                       File path to the input event points (used in the computation 
                       of cluster level of edges), must be in the same projection 
                       system with network      
    weight           : string
                       type of binary spatial weights
                       two options are allowed: Node-based, Distance-based
    distance_threshold: float
                       threshold distance value for the distance-based weight
    lisa_func        : string
                       type of LISA functions
                       three options allowed: moran, g, and g_star
    sim_method       : string
                       type of simulation methods
                       four options allowed: permutations, binomial (unconditional),
                       poisson (unconditional), multinomial (conditional)
    sim_num          : integer
                       the number of simulations
    alpha            : float
                       the significance level
    edge_weight_mode : string
                       method of calculating edge weight. Two options allowed: "multiplicative" or "additive"
    normalize_method : string
                       method of normalizing clustering values. Two options allowed: "quantile" or "linear"
    normalize        : list
                       a list of two floats indicating the normalized range for edge weights
    Examples
    --------
    
    No weights included
    
    >>> import netvoronoi as nv
    >>> nv.netvoronoi('test/streets.shp', 'test/stations.shp', 'test/netvoronoi_lincs_no_wgt.shp','ID_test', None, None,'test/crimes.shp')
    Reading network test/streets.shp: 0.032 seconds
    Reading event file : 0.031 seconds
    Local clustering level computing : 3.953 seconds
    Reading stations test/crimes.shp: 0.031 seconds
    Snapping stations to the network: 3.688 seconds
    Finding the nearest event for every node: 2.343 seconds
    Finding the nearest event for every edge (or new ones): 0.016 seconds
    Writing the result out test/netvoronoi_lincs_no_wgt.shp: 0.094 seconds
    Done
    
    ...
    """
    
    st_time = datetime.now()
    G_origin = pynet.read_network(network_file)
    G = copy.deepcopy(G_origin)
    print_info('Reading network '+network_file,st_time)
    
    #TO-DO: we may need to first split edges in G to be of equal-distance 
    
    if event_file != None:
        # G2 for lisa computation, it will modify the network inside, so better to be cautious~
        st_time = datetime.now()
        G2 = copy.deepcopy(G_origin)
        events = read_events(event_file)
        print_info('Reading event file ',st_time)
        
        st_time = datetime.now()
        event_index = snap_and_count(events,G2)
        cluster_levels,_ = lincs.lincs(G2, event_index, 0, weight, distance_threshold, lisa_func, sim_method, sim_num)
        
        # normalization
        v_arr = [v[3] for v in cluster_levels]
        if lisa_func == "moran":
            if normalize_method == "linear":
                new_arr = []
                for v in v_arr:
                    if v < -1:
                        v = -1
                    elif v > 1:
                        v = 1
                    new_arr.append( (v + 1.0)/2.0 * (normalize[1]- normalize[0]) + normalize[0] )
                v_arr = new_arr
            else:
                index_arr = sorted(enumerate(v_arr), key=operator.itemgetter(1))
                len_significant = sum ([1 if v[5]< alpha else 0 for v in cluster_levels]) 
                if len_significant > 1:
                    new_arr = [0]*len(v_arr)
                    counter = 0.0
                    for i in index_arr:
                        if cluster_levels[i[0]][5] < alpha:
                            new_arr[i[0]] = counter * (normalize[1]- normalize[0])/ (len_significant-1) + normalize[0]
                            counter += 1.0
                    v_arr = new_arr
                
        # spread cluster level values into edge length
        counter = 0
        negative_c = 0
        for v in cluster_levels:
            if v[5] < alpha:
                if edge_weight_mode == "multiplicative":
                    assert v_arr[counter]>0
                    G[v[0][0]][v[0][1]] *= v_arr[counter]
                    G[v[0][1]][v[0][0]] *= v_arr[counter]
                else:
                    G[v[0][0]][v[0][1]] += v_arr[counter]
                    G[v[0][1]][v[0][0]] += v_arr[counter]
                    
                    if G[v[0][0]][v[0][1]]<0:
                        G[v[0][0]][v[0][1]] = 0
                        G[v[0][1]][v[0][0]] = 0
                        negative_c += 1
            counter += 1
            
        print('negative count', negative_c)
        print_info('Local clustering level computing ',st_time)
    
    st_time = datetime.now()
    # read the stations in
    stations, id_spec = read_stations(station_file,id_field,add_w_field,mul_w_field)
    print_info('Reading stations '+station_file,st_time)
    
    # get a mapping for the event points and the projected ones
    st_time = datetime.now() 
    evt_snapped_dic = snap_and_split(stations, G, G_origin)
    print_info('Snapping stations to the network',st_time)
    
    # For each node in the graph, find the nearest event point 
    st_time = datetime.now() 
    distance_dic = {} #the nearest distance recorded for each node
    vertex_dic = {} #the node cover vertex_dic
    for e in stations:
        distances = pynet.dijkstras(G, evt_snapped_dic[e.id])
        for k,v in list(distances.items()):
            v = v * e.mul_wgt + e.add_wgt
            if distance_dic.get(k, 1e10) > v:
                distance_dic[k] = v
                vertex_dic[k] = e
    print_info('Finding the nearest event for every node',st_time)
    
    """
    traverse through all the edges, if the stations attached by two end nodes are 
    the same, we say this edge belongs to that event. If not, find the break point 
    and form two new sub-edges.
    """
    st_time = datetime.now() 
    edge_belong_dic = {}
    used = {}
    for k, neighbors in list(G.items()):
        assert not isinstance(k,Station)
        assert k in vertex_dic
             
        for n in neighbors:
            assert not isinstance(n,Station)
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
   
def read_stations(station_file, id_field=None, add_w_field=None,mul_w_field=None):
    """
    Read the station points from a shp file
    """
    events = []
    s = libpysal.io.open(station_file)
    dbf = libpysal.io.open(station_file[:-3] + 'dbf')
    if s.type != libpysal.cg.shapes.Point:
        raise ValueError('File is not of type Point')
    
    fields = (id_field, add_w_field, mul_w_field)
    idxs = []
    for f in fields:
        if f == None:
            idxs.append(-1)
        else:
            try:
                idxs.append(dbf.header.index(f))
            except ValueError:
                raise Exception('filed '+f+' not found in '+station_file +' !')
    
    def field_val(r,idx,default):
        if idx>=0:
            return r[idx]
        else:
            return default

    i = 0
    for g, r in zip(s,dbf):
        events.append(Station(g,field_val(r,idxs[0],i),field_val(r,idxs[1],0.0),field_val(r,idxs[2],1.0)))
        i += 1
    
    id_spec = field_val(dbf.field_spec,idxs[0],('N', 9, 0))
    return events, id_spec

def read_events(event_file):
    """
    Read event file, attributes ignored for now
    """
    points = []
    s = libpysal.io.open(event_file)
    dbf = libpysal.io.open(event_file[:-3] + 'dbf')
    if s.type != libpysal.cg.shapes.Point:
        raise ValueError('File is not of type Point')
    
    for g, r in zip(s,dbf):
        points.append(Point(g))
    return points

def snap_and_count(events, G):
    """
    Snap the event points to the nearest segments in the network. Write the number of events
    for a given edge as a new attribute in G. 
    
    The algorithm is based on the snapping routine in network.py(from PySAL)
    """
    snapper = pynet.Snapper(G)
    done = set()
    for n1 in G:
        for n2 in G[n1]:
            if (n1, n2) in done:
                continue
            dist = G[n1][n2]
            G[n1][n2] = [dist, 0]
            G[n2][n1] = [dist, 0]
            done.add((n1,n2))
            done.add((n2,n1))
    for e in events:
        snapped = snapper.snap(e)
        cur = G[snapped[0]][snapped[1]]
        cur[1] += 1
    return 1

def snap_and_split(stations, G, G_origin = None):
    """
    Snap the station points to the nearest segments in the network. The projected 
    point will break the original edge into two sub-edges. Reordering of stations 
    takes place if multiple station points are projected to the same segment. 
    
    The algorithm is based on the snapping routine in network.py(from PySAL)
    """
    evt_snapped_dic = {}
    
    snapper = pynet.Snapper(G)
    edge_dic = {}
    for e in stations:
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
        
    for key,stations in list(edge_dic.items()):
        st = key[0:2]
        end = key[2:]
        
        """
        Sort the stations based on the distance to start node.
        The sort order is decided by the two end nodes.
        """
        stations.sort(reverse = st > end)
        
        scale = 1.0
        if G_origin != None:
            scale = G[st][end]/G_origin[st][end]
        
        # remove the original edge
        del G[st][end]
        del G[end][st]
        
        size = len(stations)
        for i in range(size):
            cur_e = stations[i]
            
            G.setdefault(cur_e, {})
            
            # add the edge with previous node
            prev_point = st
            if i != 0:
                prev_point = stations[i-1]
            
            dis_st = libpysal.cg.get_points_dist(Point(prev_point), Point(cur_e)) * scale
            G[cur_e][prev_point] = dis_st
            G[prev_point][cur_e] = dis_st
            
            # if it's the last one, add the edge connecting the end node
            if i == size-1:
                aft_point = end
                dis_end = libpysal.cg.get_points_dist(Point(cur_e), Point(aft_point)) * scale
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
    if dis < 0:
        print(dis)
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
        print('filename would end with shp')
        return
    
    shp = libpysal.io.open(output_file, 'w')
    dbf = libpysal.io.open(output_file[:-3] + 'dbf', 'w')
    dbf.header = ['ID','STATION']
    dbf.field_spec = [('N', 9, 0), event_id_spec]
    
    # traverse all the edges, write in the event id and the edge geometry.
    counter = 0
    for pair,event in list(edge_belong_dic.items()):
        shp.write(Chain([Point(pair[0]), Point(pair[1])]))                    
        dbf.write([counter, event.id])
        counter += 1
        
    shp.close()
    dbf.close()

def print_info(info,cur_time = None):
    if cur_time !=None:
        delta = datetime.now() - cur_time
        combined = delta.seconds + delta.microseconds/1E6
        print(info +": "+str(combined) + " seconds")
    else:
        print(info)