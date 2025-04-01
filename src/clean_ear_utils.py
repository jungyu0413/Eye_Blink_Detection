from scipy.spatial import distance

lambda_val = 0.00001
def l_calculate_EAR(shape): # 눈 거리 계산
	A = distance.euclidean(shape[0], shape[1])
	B = distance.euclidean(shape[4], shape[5])
#	ADD = distance.euclidean(shape[2], shape[3])

	C = distance.euclidean(shape[6], shape[7])
	ear_aspect_ratio = (A+B)/2#(2.0*C)+1e-15
	
#	ear_aspect_ratio = 3*(A+B+ADD)/(C)+1e-15
	return ear_aspect_ratio



def r_calculate_EAR(shape): # 눈 거리 계산
	A = distance.euclidean(shape[0], shape[1])
	B = distance.euclidean(shape[4], shape[5])
#	ADD = distance.euclidean(shape[2], shape[3])

	C = distance.euclidean(shape[6], shape[7])
	ear_aspect_ratio = (A+B)/2#(2.0*C)+1e-15
	#ear_aspect_ratio = 3*(A+B+ADD)/(C)+1e-15
	return ear_aspect_ratio


def ld_eye(ld,det_width,det_height,det_xmin,det_ymin):
	l_eye = []
	r_eye = []
	x_left = [ld[69*2],ld[75*2],ld[70*2],ld[74*2],ld[71*2],ld[73*2],ld[68*2],ld[72*2]]
	y_left = [ld[69*2+1],ld[75*2+1],ld[70*2+1],ld[74*2+1],ld[71*2+1],ld[73*2+1],ld[68*2+1],ld[72*2+1]]

	x_right = [ld[63*2],ld[65*2],ld[62*2],ld[66*2],ld[61*2],ld[67*2],ld[64*2],ld[60*2]]
	y_right = [ld[63*2+1],ld[65*2+1],ld[62*2+1],ld[66*2+1],ld[61*2+1],ld[67*2+1],ld[64*2+1],ld[60*2+1]]

	for x1,y1,x2,y2 in zip(x_left, y_left, x_right, y_right):
		l_eye.append([int(x1*det_width)+det_xmin, int(y1*det_height)+det_ymin])
		r_eye.append([int(x2*det_width)+det_xmin, int(y2*det_height)+det_ymin])

	return l_eye, r_eye