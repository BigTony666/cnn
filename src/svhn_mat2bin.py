import numpy as np
import scipy.io
from array import array

read_input = scipy.io.loadmat('train_svhn.mat')

j=1
output_file = open('data_batch_%d.bin' % j, 'ab')

for i in range(0, 50000):

  # create new bin file
  if i>0 and i % 10000 == 0:
    output_file.close()
    j=j+1
    output_file = open('data_batch_%d.bin' % j, 'ab')
  
  # Write to bin file
  #print(dtype)
  if read_input['y'][i] == 10:
    read_input['y'][i] = 0
  read_input['y'][i].astype('uint8').tofile(output_file)
  read_input['X'][:,:,:,i].astype('uint8').tofile(output_file)
'''
for i in range(0, 5000):
  if read_input['y'][i] == 10:
    read_input['y'][i] = 0
  read_input['y'][i].astype('uint8').tofile(output_file)
'''
output_file.close()


read_input = scipy.io.loadmat('test_svhn.mat')

output_file = open('test_batch.bin', 'ab')

for i in range(0, 10000):

  if read_input['y'][i] == 10:
    read_input['y'][i] = 0
  read_input['y'][i].astype('uint8').tofile(output_file)
  read_input['X'][:,:,:,i].astype('uint8').tofile(output_file)
output_file.close()

