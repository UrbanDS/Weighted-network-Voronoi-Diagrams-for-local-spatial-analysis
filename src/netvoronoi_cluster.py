"""
A library for computing network Voronoi diagram with weighted links based on local-scale clustering analysis
The algorithm for network Voronoi diagram is originally described in Spatial Analysis along Networks, 2012, Chapter 4

The local clustering levels of network edges are incorporated. The edge weight
can be multiplicative or additive. The clustering level can either be calculated from 
kernel density estimation or local indicators of clustering such as local Moran's I

The library includes the additive weight and the multiplicative weight 
for the point events. Currently it only deals with undirected network. 

The library relies on PySAL 1.4 (or higher), particularly: network.py, kernal.py and lincs.py in the 
unreleased PySAL development version

"""
import network as pynet
import pysal.cg
from pysal.cg.shapes import Point, Chain
from datetime import datetime
import lincs
import copy
import operator
import kernel
import numpy
import os
import errno

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
        self.loc = loc
        super(Station,self).__init__(loc)
        
    def __lt__(self, other):
        if isinstance(other, Point):
            return (self.loc) < (other.loc)
        else:
            return (self.loc) < other

    def __le__(self, other):
        if isinstance(other, Point):
            return (self.loc) <= (other.loc)
        else:
            return (self.loc) <= other    
    
    def __eq__(self, other):
        try:
            return (self.id) == (other.id)
        except AttributeError:
            return False
    
    def __ne__(self, other):
        try:
            return (self.id) != (other.id)
        except AttributeError:
            return True
        
    def __gt__(self, other):
        if isinstance(other, Point):
            return (self.loc) > (other.loc)
        else:
            return (self.loc) > other

    def __ge__(self, other):
        if isinstance(other, Point):
            return (self.loc) >= (other.loc)
        else:
            return (self.loc) >= other
        
    def __add__ (self,other):
        if isinstance(other, tuple):
            return (self.loc) + other
        else:
            return (self.loc) + (other.loc)
    
    def __radd__ (self,other):
        if isinstance(other, tuple):
            return other + (self.loc)
        else:
            return (other.loc)+(self.loc)
        
    def __str__(self):
        return str(self[:])+" id: "+str(self.id) + " additive weight: " + str(self.add_wgt)+" multiplicative weight: "+ str(self.mul_wgt)

class Event(Point):
    """
    The Event point with its id

    Attributes
    ----------

    id              : Object
                      The identifier
    """
    def __init__(self, loc, id, add_wgt = 0.0, mul_wgt = 1.0):
        self.id = id
        self.add_wgt = add_wgt
        self.mul_wgt = mul_wgt
        self.loc = loc
        super(Event,self).__init__(loc)

    def __lt__(self, other):
        if isinstance(other, Point):
            return (self.loc) < (other.loc)
        else:
            return (self.loc) < other

    def __le__(self, other):
        if isinstance(other, Point):
            return (self.loc) <= (other.loc)
        else:
            return (self.loc) <= other
        
    def __eq__(self, other):
        try:
            return (self.id) == (other.id)
        except AttributeError:
            return False
    
    def __ne__(self, other):
        try:
            return (self.id) != (other.id)
        except AttributeError:
            return True
        
    def __gt__(self, other):
        if isinstance(other, Point):
            return (self.loc) > (other.loc)
        else:
            return (self.loc) > other

    def __ge__(self, other):
        if isinstance(other, Point):
            return (self.loc) >= (other.loc)
        else:
            return (self.loc) >= other
    
    def __add__ (self,other):
        if isinstance(other, tuple):
            return (self.loc) + other
        else:
            return (self.loc) + (other.loc)
        
    def __radd__ (self,other):
        if isinstance(other, tuple):
            return other + (self.loc)
        else:
            return (other.loc)+(self.loc)
        
    def __str__(self):
        return str(self[:])+" id: "+str(self.id) + " additive weight: " + str(self.add_wgt)+" multiplicative weight: "+ str(self.mul_wgt)
        
class TimeWrapper():
    """
    An wrapper for current time

    Attributes
    ----------

    time           : Object
                     current time
    """
    def __init__(self, dt):
        self.time = dt

def net_density(network_file, event_file, cell_width, bandwidth, kernel_type='quadratic',
               edge_weight_mode = "multiplicative", normalize_method = "quantile", normalize = [0.5, 1.5]):
    """
    Compute the network Voronoi diagram and write the result to a shp file.

    Parameters
    ----------
    
    network_file     : string
                       File path of the input network, must be projected
    event_file       : string  
                       File path to the input event points (used in the computation 
                       of cluster level of edges), must be in the same projection 
                       system with network      
    cell_width        : a float
                       cell width
    bandwidth        : a float
                       Kernel bandwidth
    orig_nodes       : a list of tuples
                       a tuple is the coordinate of a node that is part of the original base network
                       each tuple takes the form of (x,y)
    kernel_type      : string
                       the type of Kernel function
                       allowed values: 'quadratic', 'gaussian', 'quartic', 'uniform', 'triangular'
    edge_weight_mode : string
                       method of calculating edge weight. Two options allowed: "multiplicative" or "additive"
    normalize_method : string
                       method of normalizing clustering values. Two options allowed: "quantile" or "linear"
    normalize        : list
                       a list of two floats indicating the normalized range for edge weights
    
    Returns
    --------
    
    G_weighted       : dictionary
                       A dictionary of dictionaries like {n1:{n2:d12,...},...}
                       A planar undirected network with weighted links
    G_origin         : dictionary
                       A dictionary of dictionaries like {n1:{n2:d12,...},...}
                       A planar undirected network (the original network with euclidian distance)
    density_dic_ori  : dictionary
                       keys are node and values are their densities
                       Example: {n1:d1,n2:d2,...}
    """
    
    st_time = TimeWrapper(datetime.now())
    G_origin, orig_nodes_unmeshed = read_mesh_network(network_file,cell_width)
    print_info('Reading and meshing and meshing network '+network_file,st_time)
    
    G_mesh_copy,evt_snapped = read_snap_split_event(G_origin, event_file)
    print_info('Read and snap event points ',st_time)
    
    density_dic_ori = kernel.kernel_density(G_mesh_copy, evt_snapped, bandwidth, orig_nodes_unmeshed, kernel_type)
    print_info('Compute kernel density ',st_time)
    
    G_weighted = weighted_kde(G_origin,density_dic_ori,normalize_method,edge_weight_mode,normalize)
    print_info('Normalizing and transforming edges ',st_time)
    
    return G_weighted, G_origin, density_dic_ori

def read_mesh_network(network_file, cell_width):
    G_origin = pynet.read_network(network_file)
    orig_nodes_unmeshed = []
    for key in G_origin:
        orig_nodes_unmeshed.append(key)
    G_origin = pynet.mesh_network(G_origin, cell_width)
    return G_origin, orig_nodes_unmeshed

def read_snap_split_event(G_origin,event_file):
    events = read_events(event_file)
    G_mesh_copy = copy.deepcopy(G_origin)
    evt_snapped = snap_and_split(events,G_mesh_copy,None,False)
    return G_mesh_copy,evt_snapped

def weighted_kde(G_origin,density_dic_ori,normalize_method,edge_weight_mode,normalize):
    density_dic = copy.deepcopy(density_dic_ori)
    
    if normalize_method == "linear":
        min_d = numpy.min(density_dic.values())
        max_d = numpy.max(density_dic.values())
        density_dic_new = {}
        for n in density_dic:
            density_dic_new[n] = (density_dic[n] - min_d)/(max_d - min_d) * (normalize[1]- normalize[0]) + normalize[0]
        density_dic = density_dic_new
    else:
        index_arr = sorted(density_dic.iteritems(), key=operator.itemgetter(1))
        arr_set = set()
        for i in index_arr:
            arr_set.add(i[1])
            
        len_arr = len(arr_set)
        density_dic_new = {}
        counter = 0.0
        prev = -9999
        for i in index_arr:
            density_dic_new[i[0]] = counter * (normalize[1]- normalize[0])/ (len_arr-1) + normalize[0]
            if i[1] != prev:
                counter += 1.0
            prev = i[1]
        density_dic = density_dic_new
        
    negative_c = 0
    G_weighted = copy.deepcopy(G_origin)
    for n1 in G_weighted:
        for n2 in G_weighted[n1]:
            density = (density_dic[n1]+density_dic[n2])/2.0
            
            if edge_weight_mode == "multiplicative":
                G_weighted[n1][n2] *= density
            else:
                G_weighted[n1][n2] += density
                if G_weighted[n1][n2]<0:
                    G_weighted[n1][n2] = 0
                    negative_c += 1    
   
    if negative_c>0:
        print 'negative count: ', negative_c
    return G_weighted    
    
def net_lincs(network_file, event_file, cell_width, weight = 'Distance-based',  distance_threshold = 300, 
               lisa_func = 'moran', sim_method = 'permutations', sim_num = 99, alpha = 0.05, 
               edge_weight_mode = "multiplicative", normalize_method = "quantile", normalize = [0.5, 1.5]):
    """
    Compute the network Voronoi diagram and write the result to a shp file.

    Parameters
    ----------
    
    network_file     : string
                       File path of the input network, must be projected
    event_file       : string  
                       File path to the input event points (used in the computation 
                       of cluster level of edges), must be in the same projection 
                       system with network      
    cell_width        : a float
                       cell width
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
    
    Returns
    --------
    
    G_weighted       : dictionary
                       A dictionary of dictionaries like {n1:{n2:d12,...},...}
                       A planar undirected network with weighted links
    G_origin         : dictionary
                       A dictionary of dictionaries like {n1:{n2:d12,...},...}
                       A planar undirected network (the original network with euclidian distance)
    cluster_levels  :  list
                       A list of tuples from the lincs computation, represented as 
                       (edges_geom, e, b, Is, Zs, p_sim, qs)
    
    """
    
    st_time = TimeWrapper(datetime.now())
    G_origin, _ = read_mesh_network(network_file,cell_width)
    print_info('Reading and meshing and meshing network '+network_file,st_time)
    
    G2, event_index = read_snap_count_event(G_origin,event_file)
    print_info('Reading and snapping event file ',st_time)
    
    cluster_levels,_ = lincs.lincs(G2, event_index, 0, weight, distance_threshold, lisa_func, sim_method, sim_num)
    print_info('Normalizing and transforming edges ',st_time)
    
    G_weighted = weighted_lincs(G_origin,cluster_levels,lisa_func,alpha,normalize_method,edge_weight_mode,normalize)
    return G_weighted, G_origin, cluster_levels

def read_snap_count_event(G_origin,event_file):
    G2 = copy.deepcopy(G_origin)
    events = read_events(event_file)
    event_index = snap_and_count([events,],G2)[0]
    return G2, event_index

def weighted_lincs(G_origin,cluster_levels,lisa_func,alpha,normalize_method,edge_weight_mode,normalize):
    G_weighted = copy.deepcopy(G_origin)
    # normalization use z values rather than I
    v_arr = [v[4] for v in cluster_levels]
    if lisa_func == "moran":
        if normalize_method == "linear":
            new_arr = []
            min_v = numpy.min(v_arr)
            max_v = numpy.max(v_arr)
            for v in v_arr:
                new_arr.append( (v - min_v)/(max_v - min_v) * (normalize[1]- normalize[0]) + normalize[0] )
            v_arr = new_arr
        else:
            index_arr = sorted(enumerate(v_arr), key=operator.itemgetter(1))
            arr_set = set()
            for i in index_arr:
                if (cluster_levels[i[0]] not in arr_set) and (cluster_levels[i[0]][5] < alpha):
                    arr_set.add(i[1])
            len_arr = len(arr_set)
            if len_arr > 1:
                new_arr = [0]*len(v_arr)
                counter = 0.0
                for i in index_arr:
                    if cluster_levels[i[0]][5] < alpha:
                        new_arr[i[0]] = counter * (normalize[1]- normalize[0])/ (len_arr-1) + normalize[0]
                        counter += 1.0
                v_arr = new_arr
    
    # spread cluster level values into edge length
    counter = 0
    negative_c = 0
    for v in cluster_levels:
        if v[5] < alpha:
            if edge_weight_mode == "multiplicative":
                assert v_arr[counter]>0
                G_weighted[v[0][0]][v[0][1]] *= v_arr[counter]
                G_weighted[v[0][1]][v[0][0]] *= v_arr[counter]
            else:
                G_weighted[v[0][0]][v[0][1]] += v_arr[counter]
                G_weighted[v[0][1]][v[0][0]] += v_arr[counter]
                
                if G_weighted[v[0][0]][v[0][1]]<0:
                    G_weighted[v[0][0]][v[0][1]] = 0
                    G_weighted[v[0][1]][v[0][0]] = 0
                    negative_c += 1
        counter += 1
    
    if negative_c>0:
        print 'negative count: ', negative_c
    
    return G_weighted   

def netvoronoi(G_weighted, station_file, output_file,
               id_field=None, add_w_field=None,mul_w_field=None,G_origin=None):
    """
    Compute the network Voronoi diagram and write the result to a shp file.

    Parameters
    ----------
    
    G_weighted       : dictionary
                       A dictionary of dictionaries like {n1:{n2:d12,...},...}
                       A planar undirected network with weighted links
    G_origin         : dictionary
                       A dictionary of dictionaries like {n1:{n2:d12,...},...}
                       A planar undirected network (the original network with euclidian distance)
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

    """
    st_time = TimeWrapper(datetime.now())
    # read the stations in
    stations, id_spec = read_stations(station_file,id_field,add_w_field,mul_w_field)
    print_info('Reading stations '+station_file,st_time)
    
    # get a mapping for the event points and the projected ones
    evt_snapped = snap_and_split(stations, G_weighted, G_origin, False)
    print_info('Snapping stations to the network',st_time)
    
    # For each node in the graph, find the nearest event point 
    distance_dic = {} #the nearest distance recorded for each node
    vertex_dic = {} #the node cover vertex_dic
    counter = 0
    for e in stations:
        distances = pynet.dijkstras(G_weighted, evt_snapped[counter])
        for k,v in distances.items():
            v = v * e.mul_wgt + e.add_wgt
            if distance_dic.get(k, 1e10) > v:
                distance_dic[k] = v
                vertex_dic[k] = e
        counter += 1
    print_info('Finding the nearest event for every node',st_time)
    
    """
    traverse through all the edges, if the stations attached by two end nodes are 
    the same, we say this edge belongs to that event. If not, find the break point 
    and form two new sub-edges.
    """
    edge_belong_dic = {}
    used = {}
    no_connect_count = 0
    for k, neighbors in G_weighted.items():
#         assert not isinstance(k,Station)
        if k not in vertex_dic:
            no_connect_count += 1
            continue
        
        for n in neighbors:
#             assert not isinstance(n,Station)
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
                break_point = find_break_point(k, n, G_weighted[k][n], distance_dic[k], distance_dic[n], vertex_dic[k].mul_wgt, vertex_dic[n].mul_wgt)
                edge_belong_dic[(k,break_point)] = vertex_dic[k]
                edge_belong_dic[(break_point,n)] = vertex_dic[n]
    print_info('Finding the nearest event for every edge (or new ones)',st_time)
    print_info('Number of nodes that are not linked to the main component: '+str(no_connect_count));
    
    #write the result to the shp file
    write_edges_to_shp(edge_belong_dic, output_file, id_spec)
    print_info('Writing the result out '+output_file,st_time)
    print_info('Done')
    return edge_belong_dic
   
def read_stations(station_file, id_field=None, add_w_field=None,mul_w_field=None):
    """
    Read the station points from a shp file
    
    Parameters
    ----------
    
    station_file     : string  
                       File path to the input station points, must be in the same
                       projection system with network
    id_field         : string
                       the identifier field for the stations, created automatically 
                       if None is passed in
    add_w_field      : string
                       the additive weight field for the stations. If set to None, 
                       the additive weight will be 0 for all stations 
    mul_w_field      : string
                       the multiplicative weight field for the stations. If set to 
                       None, the multiplicative weight will be 1 for all stations
   
    Returns
    --------
    
    stations         : list
                       the Station objects
    id_spec          : tuple
                       the specification of the identifier read by PySAL
    
    """
    
    stations = []
    s = pysal.open(station_file)
    dbf = pysal.open(station_file[:-3] + 'dbf')
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
                raise Exception('filed '+f+' not found in '+station_file +' !')
    
    def field_val(r,idx,default):
        if idx>=0:
            return r[idx]
        else:
            return default

    i = 0
    for g, r in zip(s,dbf):
        stations.append(Station(g,field_val(r,idxs[0],i),field_val(r,idxs[1],0.0),field_val(r,idxs[2],1.0)))
        i += 1
    
    id_spec = field_val(dbf.field_spec,idxs[0],('N', 9, 0))
    return stations, id_spec

def read_events(event_file, id_field=None):
    """
    Read event file, attributes ignored for now
    
    Parameters
    ----------
    
    event_file       : string  
                       File path to the input event points (used in the computation 
                       of cluster level of edges), must be in the same projection 
                       system with network  
    id_field         : string
                       the identifier field for the events, created automatically 
                       if None is passed in    
                       
    Returns
    --------
    
    points           : list
                       A list of Event objects
    
    """
    
    points = []
    s = pysal.open(event_file)
    dbf = pysal.open(event_file[:-3] + 'dbf')
    if s.type != pysal.cg.shapes.Point:
        raise ValueError, 'File is not of type Point'
    
    id_idx = -1
    if id_field <>None:
        id_idx = dbf.header.index(id_field)
        
    def id_val(r,default):
        if id_idx>=0:
            return r[id_idx]
        else:
            return default
        
    i = 0
    for g, r in zip(s,dbf):
        points.append(Event(g, id_val(r,i)))
        i += 1
    return points

def snap_and_count(events_list, G):
    """
    Snap the event points to the nearest segments in the network. Write the number of events
    for a given edge as a new attribute in G. 
    
    The algorithm is based on the snapping routine in network.py(from PySAL)
    
    Parameters
    ----------
    
    events_list      : a nested list that include events at different time point  
                       a list of event 
    G                : dictionary
                       A dictionary of dictionaries like {n1:{n2:d12,...},...}
                       A planar undirected network
    
    Returns
    ----------
    
    event_index      : int
                       the event index
    """
    snapper = pynet.Snapper(G)
    done = set()
    for n1 in G:
        for n2 in G[n1]:
            if (n1, n2) in done:
                continue
            dist = G[n1][n2]
            G[n1][n2] = [dist,] + [0,]*len(events_list)
            G[n2][n1] = [dist,] + [0,]*len(events_list)
            done.add((n1,n2))
            done.add((n2,n1))
    
    counter = 1
    for events in events_list:   
        for e in events:
            snapped = snapper.snap(e)
            cur = G[snapped[0]][snapped[1]]
            cur[counter] += 1
        counter += 1
    return range(1,len(events_list)+1)

def snap_and_split(stations, G, G_origin = None, useOriginalType = True):
    """
    Snap the station points to the nearest segments in the network. The projected 
    point will break the original edge into two sub-edges. Reordering of stations 
    takes place if multiple station points are projected to the same segment. 
    
    The algorithm is based on the snapping routine in network.py(from PySAL)
    
    Parameters
    ----------
    
    stations         : list
                       the Station objects
    G                : dictionary
                       A dictionary of dictionaries like {n1:{n2:d12,...},...}
                       A planar undirected network (possibly weighted)
    G_origin         : dictionary
                       A dictionary of dictionaries like {n1:{n2:d12,...},...}
                       A planar undirected network
                       G_origin is needed because if weighted links are present, 
                       we need to know the the ratio between the weighted and the 
                       original network to produce the right split
                       
    Returns
    ----------
    
    evt_snapped      : list
                       a list of Point objects from the projection of the events to the network
                       
    """
    evt_snapped = []
    
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
        
        if useOriginalType:
            e_type = type(e)
            if e_type == Station or e_type == Event:
                point = e_type(point, e.id, e.add_wgt, e.mul_wgt)
        
        edge_dic[key].append(point)
        evt_snapped.append(point)
    
    collide_count = 0
    for key,stations in edge_dic.items():
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
            if i <> 0:
                prev_point = stations[i-1]
            
            if not useOriginalType:
                if prev_point <> cur_e: #if the geometry collides, there is no need to add it
                    dis_st = pysal.cg.get_points_dist(Point(prev_point), Point(cur_e)) * scale
                    G[cur_e][prev_point] = dis_st
                    G[prev_point][cur_e] = dis_st
                else:
                    collide_count += 1
                
                # if it's the last one, add the edge connecting the end node
                if i == size-1:
                    aft_point = end
                    if aft_point <> cur_e:
                        dis_end = pysal.cg.get_points_dist(Point(cur_e), Point(aft_point)) * scale
                        G[cur_e][aft_point] = dis_end
                        G[aft_point][cur_e] = dis_end
                    else:
                        #if the geometry collides, we replace the vertex with the station or event
                        collide_count += 1
      
            else:
                #since we now use class to represent event or station, there's no problem in having points in the same location, this will however, still 
                #causes a problem in KDE calculation
                dis_st = pysal.cg.get_points_dist(Point(prev_point), Point(cur_e)) * scale
                G[cur_e][prev_point] = dis_st
                G[prev_point][cur_e] = dis_st 
                if i == size-1:
                    aft_point = end
                    dis_end = pysal.cg.get_points_dist(Point(cur_e), Point(aft_point)) * scale
                    G[cur_e][aft_point] = dis_end
                    G[aft_point][cur_e] = dis_end
                    
    print 'collide count: ', collide_count 
    return evt_snapped

def find_break_point(p1, p2, edge_len, dp1, dp2,mul_e1=1.0,mul_e2=1.0):
    """
    Find a point on edge for two events with multiplicative weights (note that the 
    additive weight is already covered in dp1 and dp2)
    
    Parameters
    ----------
    
    p1               : Point
                       starting point
    p2               : Point
                       end point
                       A planar undirected network (possibly weighted)
    edge_len         : Number
                       length of the link
    dp1              : Number
                       the distance from the unprojected point to p1
    dp2              : Number
                       the distance from the unprojected point to p2
    mul_e1           : Number
                       multiplicative weight of p1
    mul_e1           : Number
                       multiplicative weight of p2
                                              
    Returns
    ----------
    
    p                : tuple
                       the break point
      
    """
    #distance of the breakpoint from p1
    dis = (dp2 - dp1 + edge_len * mul_e2) / (mul_e1 + mul_e2)
    if dis < 0:
        print dis
    assert dis >= 0.0
    # calculate the absolute location
    return find_point_on_edge(p1,p2,dis,edge_len)

def find_point_on_edge(p1, p2, dis_p1,edge_len):
    """
    Find a point on edge based on the distance to the starting point
    
    Parameters
    ----------
    
    p1               : Point
                       starting point
    p2               : Point
                       end point
                       A planar undirected network (possibly weighted)
    dis_p1           : Number
                       the distance from the break point to p1
    edge_len         : Number
                       length of the link
                                              
    Returns
    ----------
    
    p                : tuple
                       the break point
                       
    """
    
    if dis_p1 == 0.0:
        return p1
    dis_propotion = dis_p1/edge_len
    return tuple( (b-a)*dis_propotion + a for a,b in zip(p1,p2) )

def write_edges_to_shp(edge_belong_dic, output_file, id_spec):
    """
    write the network Voronoi computation result to a SHP file
    
    Parameters
    ----------
    
    edge_belong_dic  : dictionary
                       starting point
    output_file      : string
                       File path to the output file
    id_spec          : Number
                       the specification of the identifier read by PySAL
                       
    """
    
    make_sure_path_exists(output_file)
    if not output_file.lower().endswith('shp'):
        print 'filename would end with shp'
        return
    
    shp = pysal.open(output_file, 'w')
    dbf = pysal.open(output_file[:-3] + 'dbf', 'w')
    dbf.header = ['ID','STATION']
    dbf.field_spec = [('N', 9, 0), id_spec]
    
    # traverse all the edges, write in the event id and the edge geometry.
    counter = 0
    for pair,event in edge_belong_dic.items():
        shp.write(Chain([Point(pair[0]), Point(pair[1])]))                    
        dbf.write([counter, event.id])
        counter += 1
        
    shp.close()
    dbf.close()
    
def write_kde_to_shp(G, density_dic, output_file):
    """
    write the KDE result to a SHP file, which has the density values written for each link
    
    Parameters
    ----------
    
    G                : dictionary
                       A dictionary of dictionaries like {n1:{n2:d12,...},...}
                       A planar undirected network (possibly weighted)
    density_dic      : dictionary
                       keys are node and values are their densities
                       Example: {n1:d1,n2:d2,...}
    output_file      : string
                       File path to the output file
                       
    """    
    
    make_sure_path_exists(output_file)
    shp = pysal.open(output_file, 'w')
    dbf = pysal.open(output_file[:-3] + 'dbf', 'w')
    dbf.header = ['ID','density']
    dbf.field_spec = [('N', 9, 0), ('F', 15, 10)]
    
    counter = 0
    used = {}
    for k, neighbors in G.items():
        for n in neighbors:
            if k+n in used:
                continue
            
            used[n+k] = True
            shp.write(Chain([Point(k), Point(n)]))                    
            dbf.write([counter, (density_dic[k]+density_dic[n])/2.0])
            counter += 1
        
    shp.close()
    dbf.close()
    
def write_ilincs_to_shp(cluster_levels, output_file):
    """
    write the ilincs(local Moran's I) result to a SHP file, which has the 
    pattern,z value, i value and p value written for each link
    
    Parameters
    ----------
    
    cluster_levels  :  list
                       A list of tuples from the lincs computation, represented as 
                       (edges_geom, e, b, Is, Zs, p_sim, qs)
    output_file      : string
                       File path to the output file
                       
    """       
    make_sure_path_exists(output_file)
    shp = pysal.open(output_file, 'w')
    dbf = pysal.open(output_file[:-3] + 'dbf', 'w')
    dbf.header = ['ID','q','z','I','p']
    dbf.field_spec = [('N', 9, 0), ('N', 9, 0), ('F', 15,12), ('F', 15,12),('F', 15,12)]
    counter = 0
    for c in cluster_levels:
        shp.write(Chain([Point(c[0][0]), Point(c[0][1])]))
        if c[5] > 0.05:
            dbf.write([counter, -1, c[4], c[3],c[5]])      
        else:
            dbf.write([counter, c[6], c[4], c[3],c[5]])
        counter += 15
    
    shp.close()
    dbf.close()

def make_sure_path_exists(path):
    direc = os.path.dirname(path)
    try:
        os.makedirs(direc)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        
def print_info(info,tw = None):
    """
    utility method to print logs
    
    Parameters
    ----------
    
    info             : string
                       the note
    tw               : TimeWrapper
                       the time appended to the log
                       
    """
    
    if tw <>None:
        cur_time = tw.time
        delta = datetime.now() - cur_time
        combined = delta.seconds + delta.microseconds/1E6
        print info +": "+str(combined) + " seconds"
    else:
        print info
    if tw <>None:
        tw.time = datetime.now()
        
def get_avg_length(network):
    done = {}
    total_length = 0.0
    total_count = 0
    for n1 in network:
        for n2 in network[n1]:
            if (n1,n2) in done or (n2,n1) in done:
                continue
            total_length += network[n1][n2]
            total_count  += 1
            done[(n1,n2)] = True

    return total_length/total_count