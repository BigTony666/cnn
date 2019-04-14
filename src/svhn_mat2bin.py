import numpy as np
import scipy.io
from array import array
import os

read_input = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), 'train_svhn.mat'))

j=1
output_file = open(os.path.join(os.path.dirname(__file__), 'data_batch_%d.bin' % j), 'ab')

for i in range(0, 50000):

  # create new bin file
  if i>0 and i % 10000 == 0:
    output_file.close()
    j=j+1
    output_file = open(os.path.join(os.path.dirname(__file__), 'data_batch_%d.bin' % j), 'ab')
  
  # Write to bin file
  #print(dtype)
  if read_input['y'][i] == 10:
    read_input['y'][i] = 0
  read_input['y'][i].astype('uint8').tofile(output_file)
  read_input['X'][:,:,:,i].transpose(1,0,2).astype('uint8').tofile(output_file)
'''
for i in range(0, 5000):
  if read_input['y'][i] == 10:
    read_input['y'][i] = 0
  read_input['y'][i].astype('uint8').tofile(output_file)
'''
output_file.close()


read_input = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), 'test_svhn.mat'))

output_file = open(os.path.join(os.path.dirname(__file__), 'test_batch.bin'), 'ab')

for i in range(0, 10000):

  if read_input['y'][i] == 10:
    read_input['y'][i] = 0
  read_input['y'][i].astype('uint8').tofile(output_file)
  read_input['X'][:,:,:,i].transpose(1,0,2).astype('uint8').tofile(output_file)
output_file.close()