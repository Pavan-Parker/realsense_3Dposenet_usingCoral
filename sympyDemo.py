from  sympy import *
import numpy as np
p=Plane(Point3D(np.asarray([1,0,0])),Point3D(0,1,0),normal_vector=(0,0,1))
a=Line3D(Point3D(0,0,0),Point3D(1,1,1))
print(p.projection(Point3D(1,1,1)))
print(p.angle_between(a))
