'''
This is a library that identifies space-time interaction in the network space. Majority of the code are directly
ported from the PySAL module giddy.interaction. We had to copy the code here instead of writing 
an extension because the original code does not provide an interface for incorporating other distance types.

The methods correspond directly to the interaction module, with a 'net_' prefix added.

'''
import numpy as np
import scipy.stats as stats
from libpysal import cg
from giddy import util
from netvoronoi_cluster import snap_and_split, Event
import libpysal.weights.distance as Distance
import network as pynet
import copy
import operator
import libpysal.weights
import sys

class SpaceTimeEvents:
    """
    Method for reformatting event data stored in a shapefile for use in
    calculating metrics of spatio-temporal interaction.

    Parameters
    ----------
    path            : string
                      the path to the appropriate shapefile, including the
                      file name, but excluding the extension
    time            : string
                      column header in the DBF file indicating the column
                      containing the time stamp

    Attributes
    ----------
    n               : int
                      number of events
    x               : array
                      n x 1 array of the x coordinates for the events
    y               : array
                      n x 1 array of the y coordinates for the events
    t               : array
                      n x 1 array of the temporal coordinates for the events
    space           : array
                      n x 2 array of the spatial coordinates (x,y) for the
                      events
    time            : array
                      n x 2 array of the temporal coordinates (t,1) for the
                      events, the second column is a vector of ones
    add_wgts        : array
                      n x 1 array of the additive weights for the events
    mul_wgts        : array
                      n x 1 array of the multiplicative weights for the events
                                        
    Examples
    --------
    >>> import numpy as np
    >>> import pysal

    Read in the example shapefile data, ensuring to omit the file
    extension. In order to successfully create the event data the .dbf file
    associated with the shapefile should have a column of values that are a
    timestamp for the events. There should be a numerical value (not a
    date) in every field.

    >>> path = pysal.examples.get_path("burkitt")

    Create an instance of SpaceTimeEvents from a shapefile, where the
    temporal information is stored in a column named "T".

    >>> events = SpaceTimeEvents(path,'T')

    See how many events are in the instance.

    >>> events.n
    188

    Check the spatial coordinates of the first event.

    >>> events.space[0]
    array([ 300.,  302.])

    Check the time of the first event.

    >>> events.t[0]
    array([413])


    """
    def __init__(self, path, time_col,add_wgt_col = None,mul_wgt_col = None):
        shp = libpysal.io.open(path + '.shp')
        dbf = libpysal.io.open(path + '.dbf')

        # extract the spatial coordinates from the shapefile
        x = []
        y = []
        n = 0
        for i in shp:
            count = 0
            for j in i:
                if count == 0:
                    x.append(j)
                elif count == 1:
                    y.append(j)
                count += 1
            n += 1

        self.n = n
        x = np.array(x)
        y = np.array(y)
        self.x = np.reshape(x, (n, 1))
        self.y = np.reshape(y, (n, 1))
        self.space = np.hstack((self.x, self.y))

        # extract the temporal information from the database
        t = np.array(dbf.by_col(time_col))
        line = np.ones((n, 1))
        self.t = np.reshape(t, (n, 1))
        self.time = np.hstack((self.t, line))

        
        if add_wgt_col != None:
            self.add_wgts = np.array(dbf.by_col(add_wgt_col))
        else:
            self.add_wgts = np.zeros(n)
        
        if mul_wgt_col != None:
            self.mul_wgts = np.array(dbf.by_col(mul_wgt_col))
        else:
            self.mul_wgts = np.ones(n)
        
        # close open objects
        dbf.close()
        shp.close()
    
    def __normalize_wgts(self, wgts, normalize_method, normalize):
        if normalize_method == "linear":
            min_d = np.min(wgts)
            max_d = np.max(wgts)
            wgts_new = np.zeros(len(wgts))
            for i in range(len(wgts)):
                wgts_new[i] = (wgts[i] - min_d)/(max_d - min_d) * (normalize[1]- normalize[0]) + normalize[0]
            wgts = wgts_new
        else:
            index_arr = sorted(enumerate(wgts), key=operator.itemgetter(1))
            len_arr = len(wgts)
            new_arr = [0]*len(wgts)
            counter = 0.0
            for i in index_arr:
                new_arr[i[0]] = counter * (normalize[1]- normalize[0])/ (len_arr-1) + normalize[0]
                counter += 1.0
            wgts = new_arr
        
        return wgts
            
    def normalize_add_wgts(self, normalize_method, normalize):
        self.add_wgts = self.__normalize_wgts(self.add_wgts, normalize_method, normalize)
        
    def normalize_mul_wgts(self, normalize_method, normalize):
        self.mul_wgts = self.__normalize_wgts(self.mul_wgts, normalize_method, normalize)
        
def spat_distance_stat(events, G, distance_metric="network", spat_dis_matrix = None):
    """
    basic statistics for spatial distance 
    """
    s = events.space
    mat = spat_dis_matrix
    if mat == None:
        mat = distance_matrix(G,s,distance_metric,events.add_wgts, events.mul_wgts)
    return {'mean':np.mean(mat), 'median':np.median(mat), 'std':np.std(mat), 'skewness:': stats.skew(mat)}

def time_distance_stat(events, time_dis_matrix = None):
    """
    basic statistics for temporal distance 
    """
    t = events.t    
    mat = time_dis_matrix
    if mat == None:
        mat = cg.distance_matrix(t)
    return {'mean':np.mean(mat), 'median':np.median(mat), 'std':np.std(mat), 'skewness:': stats.skew(mat)}

def net_knox(events, delta, tau, G, distance_metric="network", permutations=99, spat_dis_matrix = None, time_dis_matrix = None):
    """
    Knox test for spatio-temporal interaction. [1]_

    Parameters
    ----------
    events          : space time events object
                      an output instance from the class SpaceTimeEvents
    delta           : float
                      threshold for proximity in space
    tau             : float
                      threshold for proximity in time
    G               : dictionary
                       A dictionary of dictionaries like {n1:{n2:d12,...},...}
                       A planar undirected network 
    distance_metric : string
                       network, manhattan, or euclidean
    permutations    : int
                      the number of permutations used to establish pseudo-
                      significance (default is 99)
    spat_dis_matrix : array-like
                      pre-computed spatial distance matrix
    time_dis_matrix : array-like
                      pre-computed temporal distance matrix
                      
    Returns
    -------
    knox_result     : dictionary
                      contains the statistic (stat) for the test and the
                      associated p-value (pvalue)
    stat            : float
                      value of the knox test for the dataset
    pvalue          : float
                      pseudo p-value associated with the statistic

    References
    ----------
    .. [1] E. Knox. 1964. The detection of space-time
       interactions. Journal of the Royal Statistical Society. Series C
       (Applied Statistics), 13(1):25-30.

    Examples
    --------
    >>> import numpy as np
    >>> import pysal

    Read in the example data and create an instance of SpaceTimeEvents.

    >>> path = pysal.examples.get_path("burkitt")
    >>> events = SpaceTimeEvents(path,'T')

    Set the random seed generator. This is used by the permutation based
    inference to replicate the pseudo-significance of our example results -
    the end-user will normally omit this step.

    >>> np.random.seed(100)

    Run the Knox test with distance and time thresholds of 20 and 5,
    respectively. This counts the events that are closer than 20 units in
    space, and 5 units in time.

    >>> result = knox(events,delta=20,tau=5,permutations=99)

    Next, we examine the results. First, we call the statistic from the
    results results dictionary. This reports that there are 13 events close
    in both space and time, according to our threshold definitions.

    >>> print(result['stat'])
    13.0

    Next, we look at the pseudo-significance of this value, calculated by
    permuting the timestamps and rerunning the statistics. In this case,
    the results indicate there is likely no space-time interaction between
    the events.

    >>> print("%2.2f"%result['pvalue'])
    0.18


    """
    n = events.n
    s = events.space
    t = events.t

    # calculate the spatial and temporal distance matrices for the events
    if spat_dis_matrix == None:
        sdistmat = distance_matrix(G,s,distance_metric,events.add_wgts, events.mul_wgts)
    else:
        sdistmat = spat_dis_matrix
    
    if time_dis_matrix == None:
        tdistmat = cg.distance_matrix(t)
    else:
        tdistmat = time_dis_matrix
        
    # identify events within thresholds
    spacmat = np.ones((n, n))
    test = sdistmat <= delta
    spacmat = spacmat * test

    timemat = np.ones((n, n))
    test = tdistmat <= tau
    timemat = timemat * test

    # calculate the statistic
    knoxmat = timemat * spacmat
    stat = (knoxmat.sum() - n) / 2
    print('stat computed: ', stat)

    # return results (if no inference)
    if permutations == 0:
        return stat
    distribution = []

    # loop for generating a random distribution to assess significance
    for p in range(permutations):
        rtdistmat = util.shuffle_matrix(tdistmat, list(range(n)))
        timemat = np.ones((n, n))
        test = rtdistmat <= tau
        timemat = timemat * test
        knoxmat = timemat * spacmat
        k = (knoxmat.sum() - n) / 2
        distribution.append(k)
        
        if p%50 == 0:
            print('permutation num: ', p)

    # establish the pseudo significance of the observed statistic
    distribution = np.array(distribution)
    exp = np.mean(distribution)
    greater = np.ma.masked_greater_equal(distribution, stat)
    count = np.ma.count_masked(greater)
    pvalue = (count + 1.0) / (permutations + 1.0)
    
    relative = 0
    if exp != 0:
        relative = stat / exp

    # return results
    knox_result = {'stat': stat, 'pvalue': pvalue, 'exp': exp, 'relative': relative, 'max': np.max(distribution), 'min': np.min(distribution)}
    return knox_result

def net_knox_grid(events, s_bandwidth, s_count, t_bandwidth, t_count, G, distance_metric="network", permutations=99, spat_dis_matrix = None, time_dis_matrix = None):
    """
    Near-repeat calculator based on the Knox test for spatio-temporal interaction. [1]_

    Parameters
    ----------
    events          : space time events object
                      an output instance from the class SpaceTimeEvents
    s_bandwidth     : float
                      spatial bandwidth
    s_count         : int
                      number of spatial bands
    t_bandwidth     : float
                      temporal bandwidth
    t_count         : int
                      number of temporal bands
    G               : dictionary
                       A dictionary of dictionaries like {n1:{n2:d12,...},...}
                       A planar undirected network 
    distance_metric : string
                       network, manhattan, or euclidean
    permutations    : int
                      the number of permutations used to establish pseudo-
                      significance (default is 99)
    spat_dis_matrix : array-like
                      pre-computed spatial distance matrix
    time_dis_matrix : array-like
                      pre-computed temporal distance matrix

    Returns
    -------
    knox_result     : dictionary
                      contains the statistic (stat) for the test and the
                      associated p-value (pvalue)
    stats            : float matrix
                      value of the knox test for the dataset
    pvalues          : float matrix
                      pseudo p-value associated with the statistic

    """
    n = events.n
    s = events.space
    t = events.t

    # calculate the spatial and temporal distance matrices for the events
    if spat_dis_matrix == None:
        sdistmat = distance_matrix(G,s,distance_metric,events.add_wgts, events.mul_wgts)
    else:
        sdistmat = spat_dis_matrix
    
    if time_dis_matrix == None:
        tdistmat = cg.distance_matrix(t)
    else:
        tdistmat = time_dis_matrix
    
    # initialize the bandwidth array
    s_list = [0]+[(x+1)*s_bandwidth for x in range(s_count)] + [sys.maxsize] # Caveat! There is a 1-meter buffer for events in the 'same location'
    t_list = [(x+1)*t_bandwidth for x in range(t_count)] + [sys.maxsize]
    
    s_list_length = len(s_list)
    t_list_length = len(t_list)
    
    # initialize the matrices
    stats = np.zeros((s_list_length, t_list_length))
    pvalues = np.zeros((s_list_length, t_list_length))
    exps = np.zeros((s_list_length, t_list_length))
    relatives = np.zeros((s_list_length, t_list_length))
    
    spacmat_list = []
    # identify events within thresholds
    for i in range(s_list_length):
        spacmat = np.ones((n, n))
        test = sdistmat <= s_list[i]
        if i > 0:
            test2 = sdistmat >= s_list[i-1]
            test = np.bitwise_and(test, test2)
            
        spacmat = spacmat * test
        spacmat_list.append(spacmat)
        
        for j in range(t_list_length):
            timemat = np.ones((n, n))
            test = tdistmat <= t_list[j]
            if j > 0:
                test2 = tdistmat >= t_list[j-1]
                test = np.bitwise_and(test, test2)
            timemat = timemat * test
            # calculate the statistic
            knoxmat = timemat * spacmat
            stats[i,j] = (knoxmat.sum() - n) / 2
        
    # loop for generating a random distribution to assess significance
    distributions = np.zeros((s_count+2, t_count+1, permutations))
    for p in range(permutations):
        rtdistmat = util.shuffle_matrix(tdistmat, list(range(n)))
        for j in range(t_list_length):
            test = rtdistmat <= t_list[j]
            timemat = np.ones((n, n))
            timemat = timemat * test
            for i in range(s_list_length):
#                spacmat = np.ones((n, n))
#                test = sdistmat <= s_list[i]
#                if i > 0:
#                    test2 = sdistmat > s_list[i-1]
#                    test = np.bitwise_and(test, test2)
#                spacmat = spacmat * test
                spacmat = spacmat_list[i]
                
                knoxmat = timemat * spacmat
                k = (knoxmat.sum() - n) / 2
                distributions[i,j,p] = k
                
        if p%100 == 0:
            print('simulation times: ', p)
                
    for i in range(s_list_length):  
        for j in range(t_list_length):  
            # establish the pseudo significance of the observed statistic
            distribution = distributions[i,j]
            exp = np.mean(distribution)
            greater = np.ma.masked_greater_equal(distribution, stats[i,j])
            count = np.ma.count_masked(greater)
            pvalue = (count + 1.0) / (permutations + 1.0)
            
            relative = 0
            if exp != 0:
                relative = stats[i,j] / exp
        
            pvalues[i,j] = pvalue
            exps[i,j] = exp
            relatives[i,j] = relative
    
    # return results
    knox_result = {'stats': stats, 'pvalues': pvalues, 'exps': exps, 'relatives': relatives}
    return knox_result

def net_mantel(events, G, distance_metric="network", permutations=99, scon=1.0, spow=-1.0, tcon=1.0, tpow=-1.0, spat_dis_matrix = None, time_dis_matrix = None):
    """
    Standardized Mantel test for spatio-temporal interaction. [2]_

    Parameters
    ----------
    events          : space time events object
                      an output instance from the class SpaceTimeEvents
    G               : dictionary
                       A dictionary of dictionaries like {n1:{n2:d12,...},...}
                       A planar undirected network 
    distance_metric : string
                       network, manhattan, or euclidean
    permutations    : int
                      the number of permutations used to establish pseudo-
                      significance (default is 99)
    scon            : float
                      constant added to spatial distances
    spow            : float
                      value for power transformation for spatial distances
    tcon            : float
                      constant added to temporal distances
    tpow            : float
                      value for power transformation for temporal distances
    spat_dis_matrix : array-like
                      pre-computed spatial distance matrix
    time_dis_matrix : array-like
                      pre-computed temporal distance matrix

    Returns
    -------
    mantel_result   : dictionary
                      contains the statistic (stat) for the test and the
                      associated p-value (pvalue)
    stat            : float
                      value of the knox test for the dataset
    pvalue          : float
                      pseudo p-value associated with the statistic

    Reference
    ---------
    .. [2] N. Mantel. 1967. The detection of disease clustering and a
    generalized regression approach. Cancer Research, 27(2):209-220.

    Examples
    --------
    >>> import numpy as np
    >>> import pysal

    Read in the example data and create an instance of SpaceTimeEvents.

    >>> path = pysal.examples.get_path("burkitt")
    >>> events = SpaceTimeEvents(path,'T')

    Set the random seed generator. This is used by the permutation based
    inference to replicate the pseudo-significance of our example results -
    the end-user will normally omit this step.

    >>> np.random.seed(100)

    The standardized Mantel test is a measure of matrix correlation between
    the spatial and temporal distance matrices of the event dataset. The
    following example runs the standardized Mantel test without a constant
    or transformation; however, as recommended by Mantel (1967) [2]_, these
    should be added by the user. This can be done by adjusting the constant
    and power parameters.

    >>> result = mantel(events, 99, scon=1.0, spow=-1.0, tcon=1.0, tpow=-1.0)

    Next, we examine the result of the test.

    >>> print("%6.6f"%result['stat'])
    0.048368

    Finally, we look at the pseudo-significance of this value, calculated by
    permuting the timestamps and rerunning the statistic for each of the 99
    permutations. According to these parameters, the results indicate
    space-time interaction between the events.

    >>> print("%2.2f"%result['pvalue'])
    0.01


    """
    n = events.n
    s = events.space
    t = events.t

    # calculate the spatial and temporal distance matrices for the events
    if spat_dis_matrix == None:
        distmat = distance_matrix(G,s,distance_metric,events.add_wgts, events.mul_wgts)
    else:
        distmat = spat_dis_matrix
    
    if time_dis_matrix == None:
        timemat = cg.distance_matrix(t)
    else:
        timemat = time_dis_matrix

    # calculate the transformed standardized statistic
    timevec = (util.get_lower(timemat) + tcon) ** tpow
    distvec = (util.get_lower(distmat) + scon) ** spow
    stat = stats.pearsonr(timevec, distvec)[0].sum()

    # return the results (if no inference)
    if permutations == 0:
        return stat

    # loop for generating a random distribution to assess significance
    dist = []
    for i in range(permutations):
        trand = util.shuffle_matrix(timemat, list(range(n)))
        timevec = (util.get_lower(trand) + tcon) ** tpow
        m = stats.pearsonr(timevec, distvec)[0].sum()
        dist.append(m)
        
        if i%50 == 0:
            print('permutation num: ', i)

    ## establish the pseudo significance of the observed statistic
    distribution = np.array(dist)
    exp = np.mean(distribution)
    greater = np.ma.masked_greater_equal(distribution, stat)
    count = np.ma.count_masked(greater)
    pvalue = (count + 1.0) / (permutations + 1.0)

    relative = 0
    if exp != 0:
        relative = stat / exp

    # return results
    mantel_result = {'stat': stat, 'pvalue': pvalue, 'exp': exp, 'relative': relative, 'max': np.max(distribution), 'min': np.min(distribution)}
    return mantel_result

def net_modified_knox(events, delta, tau, G, distance_metric="network", permutations=99, spat_dis_matrix = None, time_dis_matrix = None):
    """
    Baker's modified Knox test for spatio-temporal interaction. [1]_

    Parameters
    ----------
    events          : space time events object
                      an output instance from the class SpaceTimeEvents
    delta           : float
                      threshold for proximity in space
    tau             : float
                      threshold for proximity in time
    G               : dictionary
                       A dictionary of dictionaries like {n1:{n2:d12,...},...}
                       A planar undirected network 
    distance_metric : string
                       network, manhattan, or euclidean
    permutations    : int
                      the number of permutations used to establish pseudo-
                      significance (default is 99)
    spat_dis_matrix : array-like
                      pre-computed spatial distance matrix
    time_dis_matrix : array-like
                      pre-computed temporal distance matrix
                      
    Returns
    -------
    modknox_result  : dictionary
                      contains the statistic (stat) for the test and the
                      associated p-value (pvalue)
    stat            : float
                      value of the modified knox test for the dataset
    pvalue          : float
                      pseudo p-value associated with the statistic

    References
    ----------
    .. [1] R.D. Baker. Identifying space-time disease clusters. Acta Tropica,
       91(3):291-299, 2004


    Examples
    --------
    >>> import numpy as np
    >>> import pysal

    Read in the example data and create an instance of SpaceTimeEvents.

    >>> path = pysal.examples.get_path("burkitt")
    >>> events = SpaceTimeEvents(path,'T')

    Set the random seed generator. This is used by the permutation based
    inference to replicate the pseudo-significance of our example results -
    the end-user will normally omit this step.

    >>> np.random.seed(100)

    Run the modified Knox test with distance and time thresholds of 20 and 5,
    respectively. This counts the events that are closer than 20 units in
    space, and 5 units in time.

    >>> result = modified_knox(events,delta=20,tau=5,permutations=99)

    Next, we examine the results. First, we call the statistic from the
    results dictionary. This reports the difference between the observed
    and expected Knox statistic.

    >>> print("%2.8f"%result['stat'])
    2.81016043

    Next, we look at the pseudo-significance of this value, calculated by
    permuting the timestamps and rerunning the statistics. In this case,
    the results indicate there is likely no space-time interaction.

    >>> print("%2.2f"%result['pvalue'])
    0.11

    """
    n = events.n
    s = events.space
    t = events.t

    # calculate the spatial and temporal distance matrices for the events
    if spat_dis_matrix == None:
        sdistmat = distance_matrix(G,s,distance_metric,events.add_wgts, events.mul_wgts)
    else:
        sdistmat = spat_dis_matrix
    
    if time_dis_matrix == None:
        tdistmat = cg.distance_matrix(t)
    else:
        tdistmat = time_dis_matrix

    # identify events within thresholds
    spacmat = np.ones((n, n))
    spacbin = sdistmat <= delta
    spacmat = spacmat * spacbin
    timemat = np.ones((n, n))
    timebin = tdistmat <= tau
    timemat = timemat * timebin

    # calculate the observed (original) statistic
    knoxmat = timemat * spacmat
    obsstat = (knoxmat.sum() - n)

    # calculate the expectated value
    ssumvec = np.reshape((spacbin.sum(axis=0) - 1), (n, 1))
    tsumvec = np.reshape((timebin.sum(axis=0) - 1), (n, 1))
    expstat = (ssumvec * tsumvec).sum()

    # calculate the modified stat
    stat = (obsstat - (expstat / (n - 1.0))) / 2.0

    # return results (if no inference)
    if permutations == 0:
        return stat
    distribution = []

    # loop for generating a random distribution to assess significance
    for p in range(permutations):
        rtdistmat = util.shuffle_matrix(tdistmat, list(range(n)))
        timemat = np.ones((n, n))
        timebin = rtdistmat <= tau
        timemat = timemat * timebin

        # calculate the observed knox again
        knoxmat = timemat * spacmat
        obsstat = (knoxmat.sum() - n)

        # calculate the expectated value again
        ssumvec = np.reshape((spacbin.sum(axis=0) - 1), (n, 1))
        tsumvec = np.reshape((timebin.sum(axis=0) - 1), (n, 1))
        expstat = (ssumvec * tsumvec).sum()

        # calculate the modified stat
        tempstat = (obsstat - (expstat / (n - 1.0))) / 2.0
        distribution.append(tempstat)
        
        if p%50 == 0:
            print('permutation num: ', p)

    # establish the pseudo significance of the observed statistic
    distribution = np.array(distribution)
    exp = np.mean(distribution)
    greater = np.ma.masked_greater_equal(distribution, stat)
    count = np.ma.count_masked(greater)
    pvalue = (count + 1.0) / (permutations + 1.0)

    relative = 0
    if exp != 0:
        relative = stat / exp

    # return results
    modknox_result = {'stat': stat, 'pvalue': pvalue, 'exp': exp, 'relative': relative, 'max': np.max(distribution), 'min': np.min(distribution)}
    return modknox_result

def net_jacquez(events, k, G, distance_metric="network", permutations=99, spat_dis_matrix = None):
    """
    Jacquez k nearest neighbors test for spatio-temporal interaction. [3]_

    Parameters
    ----------
    G               : dictionary
                       A dictionary of dictionaries like {n1:{n2:d12,...},...}
                       A planar undirected network 
    events          : space time events object
                      an output instance from the class SpaceTimeEvents
    k               : int
                      the number of nearest neighbors to be searched
    G               : dictionary
                       A dictionary of dictionaries like {n1:{n2:d12,...},...}
                       A planar undirected network 
    distance_metric : string
                       network, manhattan, or euclidean
    permutations    : int
                      the number of permutations used to establish pseudo-
                      significance (default is 99)
    spat_dis_matrix : array-like
                      pre-computed spatial distance matrix

    Returns
    -------
    jacquez_result  : dictionary
                      contains the statistic (stat) for the test and the
                      associated p-value (pvalue)
    stat            : float
                      value of the Jacquez k nearest neighbors test for the
                      dataset
    pvalue          : float
                      p-value associated with the statistic (normally
                      distributed with k-1 df)

    References
    ----------
    .. [3] G. Jacquez. 1996. A k nearest neighbour test for space-time
       interaction. Statistics in Medicine, 15(18):1935-1949.


    Examples
    --------
    >>> import numpy as np
    >>> import pysal

    Read in the example data and create an instance of SpaceTimeEvents.

    >>> path = pysal.examples.get_path("burkitt")
    >>> events = SpaceTimeEvents(path,'T')

    The Jacquez test counts the number of events that are k nearest
    neighbors in both time and space. The following runs the Jacquez test
    on the example data and reports the resulting statistic. In this case,
    there are 13 instances where events are nearest neighbors in both space
    and time.

    >>> np.random.seed(100)
    >>> result = jacquez(events,k=3,permutations=99)
    >>> print result['stat']
    13

    The significance of this can be assessed by calling the p-
    value from the results dictionary, as shown below. Again, no
    space-time interaction is observed.

    >>> print("%2.2f"%result['pvalue'])
    0.21

    """
    n = events.n
    time = events.time
    space = events.space

    # calculate the nearest neighbors in space and time separately
    knns = k_neighbor_matrix(k, G, space, distance_metric, events.add_wgts, events.mul_wgts,spat_dis_matrix)
    knnt = knnW(time, k)
        
    nnt = knnt.neighbors
    nns = knns.neighbors
    knn_sum = 0

    # determine which events are nearest neighbors in both space and time
    for i in range(n):
        t_neighbors = nnt[i]
        s_neighbors = nns[i]
        check = set(t_neighbors)
        inter = check.intersection(s_neighbors)
        count = len(inter)
        knn_sum += count

    stat = knn_sum

    # return the results (if no inference)
    if permutations == 0:
        return stat

    # loop for generating a random distribution to assess significance
    dist = []
    for p in range(permutations):
        j = 0
        trand = np.random.permutation(time)
        knnt = knnW(trand, k)
        nnt = knnt.neighbors
        for i in range(n):
            t_neighbors = nnt[i]
            s_neighbors = nns[i]
            check = set(t_neighbors)
            inter = check.intersection(s_neighbors)
            count = len(inter)
            j += count

        dist.append(j)
        
        if p%50 == 0:
            print('permutation num: ', p)

    # establish the pseudo significance of the observed statistic
    distribution = np.array(dist)
    exp = np.mean(distribution)
    greater = np.ma.masked_greater_equal(distribution, stat)
    count = np.ma.count_masked(greater)
    pvalue = (count + 1.0) / (permutations + 1.0)
    
    relative = 0
    if exp != 0:
        relative = stat / exp

    # return results
    jacquez_result = {'k':k, 'stat': stat, 'pvalue': pvalue, 'exp': exp, 'relative': relative, 'max': np.max(distribution), 'min': np.min(distribution)}
    return jacquez_result

import scipy.spatial
from libpysal.cg import *

def knnW(data, k=2, p=2, ids=None):
    if issubclass(type(data), scipy.spatial.KDTree):
        kd = data
        data = kd.data
    elif type(data).__name__ == 'ndarray':
        kd = KDTree(data)
    else:
        print('Unsupported  type')

    # calculate
    nnq = kd.query(data, k=k + 1, p=p)
    info = nnq[1]
    neighbors = {}
    weights = {}
    if ids:
        idset = np.array(ids)
    else:
        idset = np.arange(len(info))
    for i, row in enumerate(info):
        row = row.tolist()
        if i in row:
            row.remove(i)
        else:
            row.pop()
        neighbors[idset[i]] = list(idset[row])
        weights[idset[i]] = [1] * len(neighbors[idset[i]])

    return libpysal.weights.W(neighbors, weights=weights, id_order=ids)

def k_neighbor_matrix(k, G, s,distance_metric, add_wgts, mul_wgts, spat_dis_matrix = None):
    """
    Network k-nearest neighbor based on distance calculations

    Parameters
    ----------
    G               : dictionary
                       A dictionary of dictionaries like {n1:{n2:d12,...},...}
                       A planar undirected network 
    s               : array
                      n x 2 array of the spatial coordinates (x,y) for the
                      events
    add_wgts        : array
                      n x 1 array of the additive weights for the events
    mul_wgts        : array
                      n x 1 array of the multiplicative weights for the events
    spat_dis_matrix : array-like
                      pre-computed spatial distance matrix
                      
    Returns
    -------
    w               : W instance
                      Weights object (pysal) with binary weights
    """
    D = spat_dis_matrix
    if D == None:
        D = distance_matrix(G,s,distance_metric, add_wgts, mul_wgts)
    
    neighbors = {}
    weights = {}
    for i in range(len(s)):
        arr = D[i]
        index_arr = sorted(enumerate(arr), key=operator.itemgetter(1))
        neighbors[i] = list(x[0] for x in index_arr[0:(k+1)])
        weights[i] = [1] * len(neighbors[i])
        
    return libpysal.weights.W(neighbors, weights=weights)
    
def distance_matrix(G, s, distance_metric, add_wgts, mul_wgts):
    """
    distance matrix

    Parameters
    ----------
    G               : dictionary
                       A dictionary of dictionaries like {n1:{n2:d12,...},...}
                       A planar undirected network 
    s               : array
                      n x 2 array of the spatial coordinates (x,y) for the
                      events
    distance_metric : string
                       network, manhattan, or euclidean
    add_wgts        : array
                      n x 1 array of the additive weights for the events
    mul_wgts        : array
                      n x 1 array of the multiplicative weights for the events
                      
    Returns
    -------
    D               : array-like
                      n by n diosmatrix, n being the number of the events
    """
    sdistmat = None
    if distance_metric == "network":
        sdistmat = network_distance_matrix(G,s,add_wgts, mul_wgts)
    elif distance_metric == "manhattan":
        sdistmat = cg.distance_matrix(s, 1.0)
    else:
        sdistmat = cg.distance_matrix(s)
    
    if distance_metric != "network":  
        for x in range(sdistmat.shape[0]):
            for y in range(sdistmat.shape[1]):
                sdistmat[x,y] = mul_wgts[x]*mul_wgts[y]*sdistmat[x,y] - (add_wgts[x]+add_wgts[y])
        
    return sdistmat
        

def network_distance_matrix(G, s, add_wgts, mul_wgts):
    """
    Network distance matrix

    Parameters
    ----------
    G               : dictionary
                       A dictionary of dictionaries like {n1:{n2:d12,...},...}
                       A planar undirected network 
    s               : array
                      n x 2 array of the spatial coordinates (x,y) for the
                      events
    add_wgts        : array
                      n x 1 array of the additive weights for the events
    mul_wgts        : array
                      n x 1 array of the multiplicative weights for the events
                      
    Returns
    -------
    D               : array-like
                      n by n diosmatrix, n being the number of the events
    """
    G = copy.deepcopy(G)
    events = []
    counter = 0
    for p in s:
        events.append(Event(p,counter, add_wgts[counter],mul_wgts[counter]))
        counter += 1
    
    evt_snapped = snap_and_split(events, G)
    
    D = np.zeros((counter, counter))
    i = 0
    for e in evt_snapped:
        distances = pynet.dijkstras(G, e)
        for k,v in list(distances.items()):
            if type(k) == Event:
                dis = v * e.mul_wgt * k.mul_wgt - (e.add_wgt + k.add_wgt)
                if dis > 0:
                    D[e.id, k.id] = dis
                else:
                    D[e.id, k.id] = 0
        
        if i%50 == 0:
            print('current event order: ', i)
        i += 1
    
    return D