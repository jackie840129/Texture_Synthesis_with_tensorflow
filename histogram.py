import scipy.interpolate
import skimage.io 
import numpy as np 
def origin_his(origin):
    his,b = np.histogram(origin,256,normed=True)
    calculate = origin.reshape(256*256,3)
    lis = []
    for x in calculate:
        n_bin = x[0]*256*256+x[1]*256+x[2]
        lis.append(n_bin)
    long = np.array(lis)
    his,b = np.histogram(long,256*256*256,normed=True)
    b = np.zeros(256*256*256+1,dtype=np.float32)
    for i in range(0,len(b)):
        b[i]=i

    cum_values = np.zeros(b.shape)
    cum_values[1:] = np.cumsum(his*np.diff(b))
    inv_cdf = scipy.interpolate.interp1d(cum_values, b,kind='nearest',bounds_error=True)
    rand = np.random.uniform(0.,1.,256*256)
    answer = inv_cdf(rand)
    ko = []
    for i in answer:
        R = i%256
        G = (i-R)%(256*256)/256
        B = (i-256*G-R)/(256*256)
        ko.append([B,G,R])
    answer = np.array(ko).reshape(256,256,3)/255.
    return answer
# r = np.asarray(uniform_hist(org_image[:,:,i].ravel()))

# r[r>cum_values.max()] = cum_values.max()    
# matched_image[:,:,i] = inv_cdf(r).reshape(org_image[:,:,i].shape)





# bins = bins/bins.sum()
# print(bins.shape)
# c = 0
# bins = np.cumsum(bins)
# print(bins.shape)
# rand = np.random.uniform(0.,1.,256*256)
'''
for i in rand:
    for j in range(0,len(bins)):
        if i < bins[j]:
            c+=1
            print("find",c)
            break;
print(c)
'''

'''

print(bins)
R_m = np.mean(origin)

R_s = np.std(origin)
print(R_m,R_s)
new = np.random.normal(R_m,R_s,(256,256,3))
N_m = np.mean(new)
N_s = np.std(new)
print(N_m,N_s)
nn = np.random.uniform(0.,1.,(256,256,3))
n_m = np.mean(nn)
n_s = np.std(nn)
print(n_m,n_s)

his = M.histogram_matching(new,origin)
his_2 = M.histogram_matching(nn,origin)
skimage.io.imshow(his_2)
skimage.io.show()
'''
