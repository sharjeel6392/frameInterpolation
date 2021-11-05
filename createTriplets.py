import numpy as np

data = np.load(r'/home/kirksalvator/Documents/Semesters/2020/Fall2020/CSCI 631 CV/Project/Data/indi_framedata.npy')
t_data = np.ndarray(shape = (299, 128, 128, 6))
o_data = np.ndarray(shape = (299, 128, 128, 3))
train = open(r'/home/kirksalvator/Documents/Semesters/2020/Fall2020/CSCI 631 CV/Project/Data/data_x.npy', 'wb')
test = open(r'/home/kirksalvator/Documents/Semesters/2020/Fall2020/CSCI 631 CV/Project/Data/data_y.npy', 'wb')
ctr = 0
for i in range(1, 299):
    t_data[ctr, :, :, 0:3] = data[i-1, :, :]
    t_data[ctr, :, :, 3:] = data[i+1, :, :]
    o_data[ctr] = data[i,:,:]
    ctr+=1
    if ctr%10 == 0:
        print(ctr)

print(t_data.shape)
print(o_data.shape)
np.save(train, t_data)
np.save(test, o_data)
