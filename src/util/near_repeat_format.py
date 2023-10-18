'''
Created on 2013-6-3

Transform a event shapefile to the format acceptable by the Near Repeat Calculator (NPC). 
Note that NPC can only deal with days in the temporal bandwidth.

'''
import pysal, pysal.cg
import datetime

def transform (event_file, date_field, ouput_file):
    s = pysal.open(event_file)
    dbf = pysal.open(event_file[:-3] + 'dbf')
    if s.type != pysal.cg.shapes.Point:
        raise ValueError, 'File is not of type Point'
    
    date_idx = -1
    if date_field <>None:
        date_idx = dbf.header.index(date_field)
        
    output = open(ouput_file,'w')
    now = datetime.datetime.now()
    for g, r in zip(s,dbf):
        output.write(str(g[0]))
        output.write(', ')
        output.write(str(g[1]))
        output.write(', ')
        delta = datetime.timedelta(days=r[date_idx])
        event_days = now + delta
        output.write(event_days.strftime('%d/%m/%Y'))
        output.write(', ')
        output.write(event_days.strftime('%Y/%m/%d'))
        output.write('\n')
    output.flush()    
    output.close()

transform('../data/pysal/crimes.shp', 'T', '../data/pysal/crimes.csv')