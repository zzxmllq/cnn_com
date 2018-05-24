import h5py
import numpy as np
def read(data_path):
 file=h5py.File(data_path,'r')
 data_array=file['data'][:]
 target_array=file['target'][:]
 # print data_array.shape,target_array.shape

 data_array_v=np.squeeze(data_array[:,0])[:,:,:,:,np.newaxis]
 data_array_l=np.squeeze(data_array[:,1])[:,:,:,:,np.newaxis]
 print data_array_v.shape
 data_array=np.concatenate([data_array_v,data_array_l],axis=4)
 target_array=target_array.reshape((-1, 32, 32, 32, 1))

 #print target_array.shape


 # data_array=np.concatenate([data_array[:,0],data_array[:,1]])

 return data_array,target_array
#
#read('/home/zzxmllq/h5_shapenet_dim32_sdf/h5_shapenet_dim32_sdf/train_shape_voxel_data0.h5')
# data=(5,5,5)
# data[0]
# data[1]
#
# data[0,0,0]
# data[:,0,0]
#
# data[:,0,:]==data[:,0]