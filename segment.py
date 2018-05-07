import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np 
import pymeanshift as pyms
import skimage.segmentation as seg
#import mahotas as mt
import cv2
from collections import Counter
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



out = np.zeros((615,624,3))

#image read
red=plt.imread('/Users/bodhisattwa/Documents/Postgrads/MTP/worldview/b5.bmp')
green=plt.imread('/Users/bodhisattwa/Documents/Postgrads/MTP/worldview/b3.bmp')
blue=plt.imread('/Users/bodhisattwa/Documents/Postgrads/MTP/worldview/b2.bmp')

#**ignore this**
#im=Image.open(r'WV/b5.bmp')
#im1=Image.open(r'WV/b3.bmp')
#im2=Image.open(r'WV/b2.bmp')


#image stack
out[:,:,0]=red
out[:,:,1]=green
out[:,:,2]=blue


#image segment
out=out.astype(np.uint8)
(segmented_img, labels_img, regions) = pyms.segment(out, spatial_radius=6,range_radius=4.5, min_density=50)
print(regions)
##image display
plt.subplot(1,2,1)
plt.imshow(out)
#plt.axis("off")
plt.title("Input Image")
plt.subplot(1,2,2)
plt.imshow(segmented_img)
#plt.axis("off")
plt.title('Segmented Image')
plt.show()
plt.imsave('/Users/bodhisattwa/Documents/Postgrads/MTP/worldview/segmented.png',segmented_img) 
plt.imshow(labels_img)
plt.colorbar()
#plt.show()


#feature extracion
l=np.unique(labels_img)
labels=labels_img.flatten()

#area
area={}
area=Counter(labels)
area=dict(area)


#perimeter
peri=np.zeros_like(l)
for (i,j),k in np.ndenumerate(labels_img):
	if i!=614 and j!=623:
		if labels_img[i-1,j-1]!=k or labels_img[i-1,j]!=k or labels_img[i-1,j+1]!=k or labels_img[i,j-1]!=k or labels_img[i,j+1]!=k or labels_img[i+1,j-1]!=k or labels_img[i+1,j]!=k or labels_img[i+1,j+1]!=k :
			peri[k]=peri[k]+1

#roundness
roundness=np.zeros_like(l)
for k in labels:
	roundness[k]=(4*3.14*area[k] )/ peri[k] 

#mean brightness
sum_b=np.zeros_like(l)
count=np.zeros_like(l)
mean_b=np.zeros_like(l)
for (i,j),k in np.ndenumerate(labels_img):
	sum_b[k]=sum_b[k]+out[i,j,0]+out[i,j,1]+out[i,j,2]
	count[k]=count[k]+3
	mean_b[k]=sum_b[k]/count[k]

#texture  **I DIDN'T USE THIS. TAKES LOT OF TIME TO EXECUTE.**
'''
img=cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
r=np.zeros_like(img)
textures=[None]*2002
for a in l:
	for (i,j),k in np.ndenumerate(labels_img):
		if k==a:
			r[i][j]=img[i][j]		
	t = mt.features.haralick(r)
	ht_mean  = t.mean(axis=0)
	textures[a]=ht_mean 
'''
#training
df=pd.read_csv('training_obs.csv',sep=',',index_col=None,header=None)
reg_ids=[]
meanb=[]
aread=[]
perid=[]
roundnessd=[]
t1=[]
t2=[]
t3=[]
t4=[]
t5=[]
t6=[]
t7=[]
t8=[]
t9=[]
t10=[]
t11=[]
t12=[]
t13=[]
for i in range(df.shape[0]):
	r_id=labels_img[df.iat[i,0]][df.iat[i,1]]
	reg_ids.append(r_id)
	meanb.append(mean_b[r_id])
	'''aread.append(area[r_id])
	perid.append(peri[r_id])
	roundnessd.append(roundness[r_id])
	t1.append(textures[r_id][0])
	t2.append(textures[r_id][1])
	t3.append(textures[r_id][2])
	t4.append(textures[r_id][3])
	t5.append(textures[r_id][4])
	t6.append(textures[r_id][5])
	t7.append(textures[r_id][6])
	t8.append(textures[r_id][7])
	t9.append(textures[r_id][8])
	t10.append(textures[r_id][9])
	t11.append(textures[r_id][10])
	t12.append(textures[r_id][11])
	t13.append(textures[r_id][12])''' 

#** I CALCULATED ALL THE FEATURES BUT USED ONLY MEAN, issliye baki sab commented hai **
list_cols=[meanb]
#,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13
list_labels=['meanb']
#,'t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13'
zipped = list(zip(list_labels, list_cols))

data=dict(zipped)
train_data=pd.DataFrame(data)
#target=pd.read_csv('class_labels_csv.csv',index_col=None,header=None)
target=df.iloc[:,-1]
clf=svm.SVC(kernel='rbf',C=8,gamma=1)
#X_train, X_test, y_train, y_test =train_test_split(train_data, target, test_size=0.3,random_state=21)
#clf.fit(X_train,y_train)
clf.fit(train_data.iloc[:,:],target)

#predicting
meanp=[]
areap=[]
perip=[]
roundnessp=[]
tp1=[]
tp2=[]
tp3=[]
tp4=[]
tp5=[]
tp6=[]
tp7=[]
tp8=[]
tp9=[]
tp10=[]
tp11=[]
tp12=[]
tp13=[]
for r in range(2006):
	meanp.append(mean_b[r])
	'''areap.append(area[r])
	perip.append(peri[r])
	roundnessp.append(roundness[r])
	tp1.append(textures[r][0])
	tp2.append(textures[r][1])
	tp3.append(textures[r][2])
	tp4.append(textures[r][3])
	tp5.append(textures[r][4])
	tp6.append(textures[r][5])
	tp7.append(textures[r][6])
	tp8.append(textures[r][7])
	tp9.append(textures[r][8])
	tp10.append(textures[r][9])
	tp11.append(textures[r][10])
	tp12.append(textures[r][11])
	tp13.append(textures[r][12])'''
l_cols=[meanp]
#tp1,tp2,tp3,tp4,tp5,tp6,tp7,tp8,tp9,tp10,tp11,tp12,tp13
l_labels=['meanp']
#,'tp1','tp2','tp3','tp4','tp5','tp6','tp7','tp8','tp9','tp10','tp11','tp12','tp13'
zipped = list(zip(l_labels, l_cols))
data=dict(zipped)
pred_data=pd.DataFrame(data)
pred_class=clf.predict(pred_data)


#accuracy
clf=svm.SVC(kernel='rbf',C=8,gamma=1)
X_train, X_test, y_train, y_test =train_test_split(train_data, target, test_size=0.6)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
	#print f1_score(y_test, y_pred, average='micro') 
	#print roc_auc_score(y_test, y_pred)
print accuracy_score(y_test,y_pred)

#classified image
cimg=np.zeros_like(labels_img)
for (i,j),k in np.ndenumerate(labels_img):
	if pred_class[k]==1:
		cimg[i,j]=0
	else:
		cimg[i,j]=255
plt.imshow(cimg)
plt.show()