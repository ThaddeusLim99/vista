class LasPointCloud:
  '''
  Container class for the .las file. Cuts down on unused fields from the
  raw .las file itself.
  '''
  def __init__(self, x, y, z, gps_time, scan_angle_rank, point_source_ID, lasfilename):
    self.x = x;
    self.y = y;
    self.z = z;
    self.gps_time = gps_time;
    self.scan_angle_rank = scan_angle_rank;
    self.point_source_ID = point_source_ID;
    self.lasfilename = lasfilename;
  pass

  # We shouldn't need getters, let alone setters since we are 
  # creating only one container object, but I did it just in case.
  def getX(self):
    return self.x;
  def getY(self):
    return self.y;
  def getZ(self):
    return self.z;
  def getGPSTime(self):
    return self.gps_time;
  def getScanAngleRank(self):
    return self.scan_angle_rank;
  def getPointSourceID(self):
    return self.point_source_ID;
  def getLasFileName(self):
    return self.lasfilename;

  pass