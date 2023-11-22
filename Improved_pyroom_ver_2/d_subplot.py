import numpy as np
import scipy
import sofa
from scipy.fft import fft,fftfreq,ifft
import math
import matplotlib.pyplot as plt
import collections
from scipy.spatial import KDTree
from numpy import linalg as LA
from scipy.signal import decimate
from scipy.signal import find_peaks

#Scipy function scipy.special.sph_harm works on angles that are in radians.


def cal_sph_basis(theta, phi, degree,no_of_nodes):  # theta_target,phi_target
    Ysh = np.zeros((len(theta), (degree + 1) ** 2),dtype=np.complex_)

    print("Ysh Shape ",Ysh.shape)
    # Ysh_tar=np.zeros(len(theta_target),(degree+1)**2)
    ny0=1
    for j in range(no_of_nodes):
        for i in range(degree + 1):
            m = np.linspace(0, i, int((i - 0) / 1. + 1), endpoint=True,dtype=int)
            sph_vals = [scipy.special.sph_harm(order, i, theta[j], phi[j]) for order in m]
            cal_index_Ysh_positive_order = (ny0 + m) - 1

            Ysh[j, cal_index_Ysh_positive_order] = sph_vals
            if i > 0:
                m_neg = np.linspace(-i, -1, int((-1 - -i) / 1. + 1), endpoint=True,dtype=int)
                sph_vals_neg = [scipy.special.sph_harm(order, i, theta[j], phi[j]) for order in m_neg]
                cal_index_Ysh_negative_order = (ny0 + m_neg) - 1

                Ysh[j, cal_index_Ysh_negative_order] = sph_vals_neg

            # Update index for next degree
            ny0 = ny0 + 2 * i + 2
        ny0=1
    return Ysh

def calculation_pinv_voronoi_cells(Ysh,theta_16,no_of_lat):

    res = (theta_16[:-1] + theta_16[1:]) / 2
    res = np.insert(res, len(res), np.pi)
    res = np.insert(res, 0, 0)

    w = -np.diff(np.cos(res))

    w_ = np.tile(w, no_of_lat) #Repeat matrice n times

    w_ = np.diag(w_) #Diagnol of the matrice (no_of_nodes * no_of_nodes)

    Ysh_tilda = np.matmul(w_, Ysh)

    Ysh_tilda_inv = np.linalg.pinv(Ysh_tilda, rcond=1e-2) #rcond is inverse of the condition number

    print("Condition Number Psuedo Inverse ",np.linalg.cond(Ysh_tilda_inv),Ysh_tilda_inv.shape)

    return Ysh_tilda_inv,w_


def fibonacci_sphere(samples):

    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        #r,theta,phi=asSpherical([x,y,z])

        points.append((x, y, z))


    return points

def sph2cart(phi,theta,r):

    x=np.cos(phi)*np.sin(theta)
    y=np.sin(phi)*np.sin(theta)
    z=np.cos(theta)

    return x,y,z

class interpolate_sofa_files():

    def __init__(self,deg,source=False,id=2,freq_dependent=True,azimuth_mic=0,colatitude_mic=0,mic_type=0):

        self.freq_dependent=freq_dependent

        self.points = np.array(fibonacci_sphere(samples=1200))  # Make a fibonacci sphere with 1000 points

        self.phi_fibo, self.theta_fibo, self.r_fibo = self.convert_fibonacci_sphere_radians(self.points)

        print("First conversion Cart Points to Sph Fibo",np.min(self.phi_fibo),np.max(self.phi_fibo),np.min(self.theta_fibo),np.max(self.theta_fibo))



        self.stack_phi_theta_fibo = np.hstack((self.phi_fibo.reshape(-1, 1), self.theta_fibo.reshape(-1, 1)))

        #print(self.stack_phi_theta_fibo.shape)

        self.nn_kd_tree = KDTree(self.stack_phi_theta_fibo)


        # Open sofa database

        self.id = id

        self.phi_src,self.theta_src,self.r_src,self.theta_16,self.rcv_msr_ir,self.no_of_nodes,self.samples_size_ir,self.no_of_lat=self.open_sofa_database(source)
        print("SOFA file open range sph coordinates ",np.min(self.phi_src), np.max(self.phi_src), np.min(self.theta_src), np.max(self.theta_src))
        print(self.samples_size_ir)
        self.nn_kd_tree_original_grid = KDTree(np.hstack((self.phi_src.reshape(-1,1),self.theta_src.reshape(-1,1))))

        self.degree = deg  # Specify spherical harmonics degree

        self.source=source


        # All the computations are in radians for phi = range (0, 2*np.pi) and theta = range (0, np.pi)
        # Just for numpy, the function accepts phi and theta in the opposite way
        self.theta_src_np = self.phi_src
        self.phi_src_np = self.theta_src
        self.Ysh = cal_sph_basis(self.theta_src_np, self.phi_src_np, self.degree, self.no_of_nodes)  # Calculate sph harmonics for the defined grid

        self.azimuth_mic = azimuth_mic
        self.colatitude_mic = colatitude_mic

        if not(self.freq_dependent):
            self.samples_size_ir = 128
            self.freq_independent_gain = 0
            self.mic_type = mic_type
            self.debug_mic_orientation(self.azimuth_mic, self.colatitude_mic)



        self.freq_angles_fft = np.zeros((self.no_of_nodes, self.samples_size_ir ), dtype=np.complex_) #self.samples_size_ir//2



        self.expand_whole_orginal_grid=0

        # Calculate spherical basis for the target grid (fibonacci grid)
        self.theta_fibo_np = self.phi_fibo
        self.phi_fibo_np = self.theta_fibo
        self.Ysh_fibo = cal_sph_basis(self.theta_fibo_np, self.phi_fibo_np, self.degree, 1200)


        #calculate pinv and voronoi cells for least square solution for the whole grid

        self.Ysh_tilda_inv,self.w_=calculation_pinv_voronoi_cells(self.Ysh,self.theta_16,self.no_of_lat)
        self.full_expanded_target_grid=0



        #Rotate the target grid (Fibonacci grid)

        n_c = (self.colatitude_mic)   #1.483
        n_a = (self.azimuth_mic)   #1.0471
        R_y = np.array([[np.cos(n_c), 0, np.sin(n_c)], [0, 1, 0], [-np.sin(n_c), 0, np.cos(n_c)]])
        R_z = np.array([[np.cos(n_a), -np.sin(n_a), 0], [np.sin(n_a), np.cos(n_a), 0], [0, 0, 1]])
        res = np.matmul(R_z,R_y)


        self.rotated_fibo_points = np.matmul(res, self.points.T)

        #print("Rotated Fibo Points",self.rotated_fibo_points.shape)

        self.rotated_fibo_phi,self.rotated_fibo_theta,self.rotated_fibo_r=self.convert_fibonacci_sphere_radians(self.rotated_fibo_points.T)
        self.nn_kd_tree_rotated_fibo_grid = KDTree(np.hstack((self.rotated_fibo_phi.reshape(-1, 1), self.rotated_fibo_theta.reshape(-1, 1))))
        print("Second Conversion Rotated Cart Points To Sph FIBO ",np.min(self.rotated_fibo_phi),np.max(self.rotated_fibo_phi),np.min(self.rotated_fibo_theta),np.max(self.rotated_fibo_theta))

        x,y,z=sph2cart(self.phi_src,self.theta_src,self.r_src) #Spherical to cart 1
        print("Second Conversion sph sofa to cart for rotation")
        cart_sph=np.empty((3,self.theta_src.shape[0]))
        cart_sph[0,:]=x
        cart_sph[1,:]=y
        cart_sph[2,:]=z


        self.rotated_sofa_points = np.matmul(res, cart_sph)
        self.rotated_sofa_phi,self.rotated_sofa_theta,self.rotated_sofa_r=self.convert_fibonacci_sphere_radians(self.rotated_sofa_points.T)
        self.nn_kd_tree_rotated_sofa_grid = KDTree(
            np.hstack((self.rotated_sofa_phi.reshape(-1, 1), self.rotated_sofa_theta.reshape(-1, 1))))
        print("Third Conversion Rotated Cart Points To Sph SOFA ", np.min(self.rotated_sofa_phi),
              np.max(self.rotated_sofa_phi), np.min(self.rotated_sofa_theta), np.max(self.rotated_sofa_theta))



    def open_sofa_database(self,source=False):
        # Open DirPat database
        if source:
            file_sofa = sofa.Database.open("/home/psrivast/Téléchargements/LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa")

            #Receiver positions

            rcv_pos=file_sofa.Receiver.Position.get_values()

            # CHEAP HACCCKK SPECIFICALLY FOR DIRPAT

            rcv_pos_RS = np.reshape(rcv_pos, [36, 15, 3])

            rcv_pos = np.swapaxes(rcv_pos_RS, 0, 1).reshape([540, -1])

            ###########################

            # Get impulse responses from all the measurements

            IR_S = file_sofa.Data.IR.get_values()

            # Look for source of specific type requested by user"

            rcv_msr = IR_S[self.id, :, :]  # First receiver #Shape ( no_sources * no_measurement_points * no_samples_IR)
            #print(rcv_msr.shape)
            #downsample the fir filter.

            rcv_msr=decimate(rcv_msr,int(round(file_sofa.Data.SamplingRate.get_values()[0]/16000)),axis=-1)

            #pad_zero=np.zeros((rcv_msr.shape[0],1024))

            #pad_zero[:,:rcv_msr.shape[1]]=rcv_msr

            #rcv_msr=pad_zero

            #print("source-shape",rcv_msr.shape)

            # no_of_nodes = src_sofa.Dimensions.M  # Number of measurement points

            no_of_nodes = 540

            #samples = file_sofa.Dimensions.N  # Samples per IR
            samples=rcv_msr.shape[1] #Number of samples changed after downsampling the FIR filter
            #print(samples)

            #samples=256

            # All measurements should be in = radians phi [0,2*np.pi] , theta [0,np.pi]

            phi_rcv = rcv_pos[:, 0]
            theta_rcv = rcv_pos[:, 1]
            r_rcv = rcv_pos[:, 2]

            #Calculate no of latitudes and longitudes in the grid

            k = list(collections.Counter(theta_rcv).values())

            no_of_lat = k[0]

            no_of_long = len(k)

            theta_16 = np.array([theta_rcv[i] for i in range(len(theta_rcv)) if i % no_of_lat == 0])
            #print(file_sofa.Data.SamplingRate.get_values())

            return phi_rcv, theta_rcv, r_rcv, theta_16, rcv_msr, no_of_nodes, samples, no_of_lat
        else:
            file_sofa = sofa.Database.open("/home/psrivast/Téléchargements/AKG_c480_c414_CUBE.sofa")
            #Source positions

            src_pos = file_sofa.Source.Position.get_values()

            # CHEAP HACCCKK SPECIFICALLY FOR DIRPAT

            src_pos_RS = np.reshape(src_pos, [30, 16, 3])

            src_pos = np.swapaxes(src_pos_RS, 0, 1).reshape([480, -1])

            ###########################

            # Get impulse responses from all the measurements

            IR_S = file_sofa.Data.IR.get_values()

            #print("Sampling Rate for receivers ",file_sofa.Data.SamplingRate.get_values()[0])
            # Look for receiver of specific type requested by user"

            rcv_msr = IR_S[:, self.id, :]  # First receiver #Shape (no_measurement_points * no_receivers * no_samples_IR)



            rcv_msr=decimate(rcv_msr,int(round(file_sofa.Data.SamplingRate.get_values()[0]/16000)),axis=-1)



            #pad_zero = np.zeros((rcv_msr.shape[0], 128))

            #pad_zero[:, :rcv_msr.shape[1]] = rcv_msr

            #rcv_msr = pad_zero


            no_of_nodes = file_sofa.Dimensions.M  # Number of measurement points

            #samples = file_sofa.Dimensions.N  # Samples per IR
            samples=rcv_msr.shape[1]

            #print(samples)

            # All measurements should be in = radians phi [0,2*np.pi] , theta [0,np.pi]

            phi_src = src_pos[:, 0]
            theta_src = src_pos[:, 1]
            r_src = src_pos[:, 2]

            k = list(collections.Counter(theta_src).values())

            no_of_lat = k[0]

            no_of_long = len(k)

            theta_16 = np.array([theta_src[i] for i in range(len(theta_src)) if i % no_of_lat == 0])

            return phi_src, theta_src, r_src, theta_16, rcv_msr, no_of_nodes, samples, no_of_lat




    def convert_fibonacci_sphere_radians(self,points):
        # Convert cartesian coordinates into spherical coordinates (radians)

        r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]) + np.square(points[:, 2]))

        theta_fibo = np.arccos((points[:, 2] / r))

        phi_fibo=np.arctan2(points[:, 1], points[:, 0])



        #phi_fibo += np.pi  # phi_fibo was in range of [-np.pi,np.pi]

        return phi_fibo,theta_fibo,r


    def cal_spherical_coeffs_grid(self):

        #Take n-point fft of the IR's on the grid

        #freq_angles_fft = np.zeros((self.no_of_nodes, self.samples_size_ir// 2), dtype=np.complex_)

        if self.freq_dependent:
            for i in range(self.no_of_nodes):
                self.freq_angles_fft[i, :] = fft(self.rcv_msr_ir[i, :])  # [:(self.samples_size_ir // 2)] #self.samples_size_ir//2
            print("freq angle fft",self.freq_angles_fft.shape)
        else:
            unit_v = np.array(
                [
                    np.cos(0) * np.sin(0),
                    np.sin(0) * np.sin(0),
                    np.cos(0),
                ]
            )
            abs_gain = 1.0
            mic_type = self.mic_type
            new_array = np.zeros((3, self.theta_src.shape[0]))
            x_cart_src, y_cart_src, z_cart_src = sph2cart(self.phi_src, self.theta_src, self.r_src)
            new_array[0, :] = x_cart_src
            new_array[1, :] = y_cart_src
            new_array[2, :] = z_cart_src
            # Formula for gain
            resp = abs_gain * mic_type + (1 - mic_type) * np.matmul(
                unit_v, new_array
            )
            td_filter = np.zeros((self.theta_src.shape[0], self.samples_size_ir))
            td_filter[:, 0] = resp
            self.freq_angles_fft = np.fft.fft(td_filter, axis=1)

        g_tilda = np.matmul(self.w_, self.freq_angles_fft)

        #Shape w_ : (540*540) or (480*480)
        #Shape g_tilda : (480*256),(480*128),(540*2048),(540*1048),(540,683)

        gamma_full_scale = np.matmul(self.Ysh_tilda_inv, g_tilda) #Coeffs for every freq band , for all the nodes present in the sphere

        #Shape gamma_full_scale : ((n+1)^2 * 256 | 128 | 2048 | 1024 | 683) Coeffs for every frequency bin

        #Select coeffs for particular frequency for plotting purpose
        #For sources select between (0-1023), receivers (0-128)

        freq_bin=20
        gamma = gamma_full_scale[:, freq_bin] #[:(self.samples_size_ir//2)]

        #Shape gamma : ((n+1)^2) Coeffs for a particular frequency bin .

        return gamma_full_scale,gamma,freq_bin

    def interpolate(self,gamma_full_scale,gamma):

        exp_orginal_grid = np.matmul(self.Ysh, gamma) #Shape Ysh:(480|540 *(n+1)^2)  gamma:(n+1)^2 {For a particular freq bin} Output : {480 or 540}

        exp_target_grid = np.matmul(self.Ysh_fibo, gamma) #Shape Ysh:(1000 *(n+1)^2)  gamma:(n+1)^2 {For a particular freq} Output : {1000}

        exp_whole_target_grid=np.matmul(self.Ysh_fibo,gamma_full_scale)

        self.full_expanded_target_grid=exp_whole_target_grid
        self.expand_whole_orginal_grid=np.matmul(self.Ysh,gamma_full_scale) #Shape Ysh:(480|540 *(n+1)^2) gamma_full_scale: ((n+1)^2,2048|256 or 1024|128 )
        exp_whole_grid=self.expand_whole_orginal_grid

        #Shape exp_whole_grid : (480,256),(480,128),(540,2048),(540,1024)

        #plt.plot(np.arange(128),self.freq_independent_gain_fibo[1000,:])
        #plt.plot(np.arange(128), np.abs(self.full_expanded_target_grid[1000, :]))
        #plt.show()

        return exp_orginal_grid,exp_target_grid,exp_whole_grid,exp_whole_target_grid


    def debug_mic_orientation(self,azimuth_m,colatitude_m):


        unit_v_= np.array(
            [
                np.cos(azimuth_m) * np.sin(colatitude_m),
                np.sin(azimuth_m) * np.sin(colatitude_m),
                np.cos(colatitude_m),
            ]
        )
        abs_gain=1.0
        mic_type=self.mic_type
        new_array_=np.zeros((3,1200))

        new_array_[0,:]=self.points[:,0]
        new_array_[1, :] = self.points[:,1]
        new_array_[2, :] = self.points[:,2]

        #Formula for gain
        resp = abs_gain * mic_type + (1 - mic_type) * np.matmul(
            unit_v_, new_array_
        )

        td_filter=np.zeros((1200,self.samples_size_ir))
        td_filter[:,0]=resp
        fd_filter=np.fft.fft(td_filter,axis=1)

        self.freq_independent_gain_fibo=fd_filter


    def plot(self,exp_original_grid,exp_target_grid,fibo_debug=0,freq_bin=0):



        # Convert to cartesian coordiante
        x_src = self.r_src * np.cos(self.phi_src) * np.sin(self.theta_src)
        y_src = self.r_src * np.sin(self.phi_src) * np.sin(self.theta_src)
        z_src = self.r_src * np.cos(self.theta_src)
        print("Conversion of SOFA sph to cart for plotting",np.min(self.phi_src),np.max(self.phi_src),np.min(self.theta_src),np.max(self.theta_src))

        # Convert to cartesian coordinates of fibonacci grid for checking purpose
        x_src_fibo = self.r_fibo * np.cos(self.phi_fibo) * np.sin(self.theta_fibo)
        y_src_fibo = self.r_fibo * np.sin(self.phi_fibo) * np.sin(self.theta_fibo)
        z_src_fibo = self.r_fibo * np.cos(self.theta_fibo)
        print("Conversion of FIBO sph to cart for plotting",np.min(self.phi_fibo),np.max(self.phi_fibo),np.min(self.theta_fibo),np.max(self.theta_fibo))


        x_src_fibo_ = self.rotated_fibo_r * np.cos(self.rotated_fibo_phi) * np.sin(self.rotated_fibo_theta)
        y_src_fibo_ = self.rotated_fibo_r * np.sin(self.rotated_fibo_phi) * np.sin(self.rotated_fibo_theta)
        z_src_fibo_ = self.r_fibo * np.cos(self.rotated_fibo_theta)

        print("Conversion of Rotated FIBO sph to cart for plotting",np.min(self.rotated_fibo_phi),np.max(self.rotated_fibo_phi),np.min(self.rotated_fibo_theta),np.max(self.rotated_fibo_theta))

        fg = plt.figure(figsize=plt.figaspect(0.5))


        ax1 = fg.add_subplot(1,5,1,projection='3d')
        ax_=ax1.scatter(x_src, y_src, z_src, c=np.abs(self.freq_angles_fft[:,freq_bin]),s=100)
        ax1.set_title("Mag|STFT| of one of the SOFA Files")
        fg.colorbar(ax_, shrink=0.5, aspect=5)

        ax2 = fg.add_subplot(1,5,2,projection='3d')
        ax_1=ax2.scatter(x_src, y_src, z_src, c=np.abs(exp_original_grid),s=100)
        ax2.set_title("Expanded Initial Grid")
        fg.colorbar(ax_1, shrink=0.5, aspect=5)


        ax3 = fg.add_subplot(1,5,4,projection='3d')
        ax_2 = ax3.scatter(self.rotated_fibo_points[0,:], self.rotated_fibo_points[1,:],
                           self.rotated_fibo_points[2,:], c=np.abs(exp_target_grid), s=50)

        #ax_2=ax3.scatter(self.rotated_fibo_points[0,:], self.rotated_fibo_points[1,:], self.rotated_fibo_points[2,:], c=np.abs(exp_target_grid),s=50)
        ax3.set_title("Interp And Rot Target Grid Based On Analytic ")
        fg.colorbar(ax_2, shrink=0.5, aspect=5)



        if freq_dependent:

            ax4 = fg.add_subplot(1, 5, 3, projection='3d')
            ax_3 = ax4.scatter(self.points[:,0], self.points[:,1], self.points[:,2], c=np.abs(exp_target_grid), s=50) #fibo_debug
            ax4.set_title("Interpolated Fibo Grid  ")
            fg.colorbar(ax_3, shrink=0.5, aspect=5)

        else:
            ax4 = fg.add_subplot(1, 5, 3, projection='3d')
            ax_3 = ax4.scatter(x_src_fibo, y_src_fibo, z_src_fibo, c=fibo_debug, s=50)  # fibo_debug
            ax4.set_title(" Rotated Analytic Formula ")
            fg.colorbar(ax_3, shrink=0.5, aspect=5)


        x,y,z=sph2cart(self.rotated_sofa_phi,self.rotated_sofa_theta,self.rotated_sofa_r)

        ax5 = fg.add_subplot(1, 5, 5, projection='3d')
        ax_4 = ax5.scatter(self.rotated_sofa_points[0,:], self.rotated_sofa_points[1,:], self.rotated_sofa_points[2,:], c=np.abs(self.freq_angles_fft[:,freq_bin]), s=50)  # fibo_debug
        ax5.set_title("Rot sofa Grid on Analytic Formula ")
        fg.colorbar(ax_4, shrink=0.5, aspect=5)

        fg.tight_layout(pad=3.0)
        plt.show()


        # Plot scatter
        '''
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))

        pl0 = axs[0].scatter(np.rad2deg(self.phi_src), np.rad2deg(self.theta_src), c=np.abs(self.freq_angles_fft[:, freq_bin]))
        axs[0].set_title("Mag|STFT|")
        fig.colorbar(pl0, ax=axs[0])

        pl0 = axs[1].scatter(np.rad2deg(self.phi_src), np.rad2deg(self.theta_src), c=np.abs(exp_original_grid))
        axs[1].set_title("Mag|Complex Gain| Expanded Initial Grid")
        fig.colorbar(pl0, ax=axs[1])

        pl0 = axs[2].scatter(np.rad2deg(self.phi_fibo), np.rad2deg(self.theta_fibo), c=np.abs(exp_target_grid))
        axs[2].set_title("Mag|Complex Gain| Target Grid")
        fig.colorbar(pl0, ax=axs[2])

        fig.tight_layout(pad=5.0)

        plt.savefig("Receiver_dir.png")
        '''

    def neareast_neighbour(self,azimuth,colatitude):

        longitude=azimuth#+np.pi
        latitude=colatitude
        #dd,ii=self.nn_kd_tree.query([longitude,latitude])

        dd, ii = self.nn_kd_tree_rotated_fibo_grid.query([longitude, latitude])  #Query on the rotated set of points

        #plt.clf()
        #dd,ll=self.nn_kd_tree_original_grid.query([longitude,latitude])
        #plt.plot(np.arange(256),ifft(self.expand_whole_orginal_grid[ll,:]),label="Original grid")
        #plt.plot(np.arange(256), ifft(self.full_expanded_target_grid[ii, :]),color="orange",label="Target grid")
        #plt.title("Ir on original grid and on interpolated target grid")
        #plt.legend()
        #plt.show()

        #if self.freq_dependent:


        #print(dd, ii, self.full_expanded_target_grid.shape)

        #return self.freq_independent_gain_fibo[ii,:]
        return self.full_expanded_target_grid[ii,:]
        #return self.freq_angles_fft[ii,:]
        #else:
        #    print("To check if 25 IS values are same",self.freq_independent_gain[ii,0])
        #    return self.freq_independent_gain[ii,:]
        #print("Gain for the point "+str(longitude)+" "+str(latitude)+"at freq: "+str(fftfreq(self.samples_size_ir,d=1/44100)[freq_bin])+" is: "+str(self.full_expanded_target_grid[ii,freq_bin]))

#########
#Important Getting less points when plotting for receivers.



#degree=[2,4,6,8,10,12,14,16,18,20,22,24,26]
#degree=[4,8,12,16,20,24]
#degree=[2,4,6,8]
degree=[8]
id_=[0,2,4,5]
colors=["red","blue","green","magenta","purple","cyan","orange","gold","teal","orchid","darkorange","brown","royalblue"]
ll=[]


rec_id=[4]
#rec_id=[0,1,2,3,4]
#src_id=[0,1,2,3,4,5,6,7,8,9,10,11]

#src_id=[2]

#fig, axs = plt.subplots(2, 2,figsize=(10,10))
#25,35
fg = plt.figure(figsize=plt.figaspect(0.2))
s=[]
q_=0
freq_dependent=True
#Pattern=["Cardioid","Omnidirectional","Wide Cardioid","Hyper Cardioid","Figure Of Eight"]
Pattern=["Genelec 8020","Bruel and Kjaer Dummy Head","Neumann KH120A","Yamaha DXR8"]


for i in id_:

    l=interpolate_sofa_files(12,source=True,id=i,freq_dependent=freq_dependent,azimuth_mic=np.deg2rad(0),colatitude_mic=np.deg2rad(0),mic_type=0) #When changing the source keep check of the freq bin , and receiver, source type by their index.

    gamma_full_scale,gamma,freq_bin=l.cal_spherical_coeffs_grid()

    #print(gamma_full_scale.shape)

    exp_original_grid_freqbin, exp_target_grid_freqbin,exp_whole_original_grid,exp_whole_target_grid =l.interpolate(gamma_full_scale,gamma)

    x_src = l.r_src * np.cos(l.phi_src) * np.sin(l.theta_src)
    y_src = l.r_src * np.sin(l.phi_src) * np.sin(l.theta_src)
    z_src = l.r_src * np.cos(l.theta_src)

    #if not(freq_dependent):
    #    l.plot(exp_original_grid_freqbin, exp_whole_target_grid[:, freq_bin], fibo_debug=l.freq_independent_gain_fibo[:, freq_bin],freq_bin=freq_bin)
    #else:
    #    l.plot(exp_original_grid_freqbin, exp_whole_target_grid[:, freq_bin],fibo_debug=0,
    #           freq_bin=freq_bin)

    print("freq angle fft",l.freq_angles_fft.shape)

    axs = fg.add_subplot(2, 4, q_+1, projection='3d')
    s = axs.scatter(x_src,y_src,z_src,c=np.abs(l.freq_angles_fft[:,300]))
    axs.set_title(Pattern[q_])
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_zticks([])

    fg.colorbar(s, shrink=0.5, aspect=5)

    axs_ = fg.add_subplot(2, 4, (q_+1)+4, projection='3d')

    s_ = axs_.scatter(l.rotated_fibo_points[0,:] , l.rotated_fibo_points[1,:], l.rotated_fibo_points[2,:], c=np.abs(exp_whole_target_grid[:,300]))
    axs_.set_xticks([])
    axs_.set_yticks([])
    axs_.set_zticks([])

    #axs_.set_title("Interpolated on Fibo sphere")


    fg.colorbar(s_, shrink=0.5, aspect=5)
    '''
    s = axs[0,q_].scatter(np.rad2deg(l.phi_fibo), np.rad2deg(l.theta_fibo), c=np.abs(exp_whole_target_grid[:,300]))
    o=plt.colorbar(s, ax=axs[0,q_])
    axs[0,q_].set_title("Interpolated pattern on fibo grid "+Pattern[q_])
    o.set_label("|STFT|")
    dd,ii=l.nn_kd_tree_original_grid.query([l.phi_src[45],l.theta_src[73]])
    dd_,ii_=l.nn_kd_tree_rotated_fibo_grid.query([l.phi_src[45], l.theta_src[73]])
    axs[1,q_].plot(np.arange(l.freq_angles_fft.shape[1]), ifft(l.freq_angles_fft[ii,:]),label="FIR on original grid")
    axs[1,q_].plot(np.arange(l.freq_angles_fft.shape[1]), ifft(exp_whole_target_grid[ii_, :]),label="FIR on interpolated grid")
    axs[1,q_].legend()
    '''
    q_+=1
    #o.set_label("dB")

#fg.tight_layout(pad=0.5)

#plt.title("Directivity Plot Of SOFA Mics Frequency Bin = 5000 Hz ")
plt.show()
    #l.plot(exp_original_grid_freqbin, l.freq_independent_gain_fibo[:, freq_bin] , freq_bin)



'''
    print("for a particular freq bin", exp_original_grid_freqbin.shape, exp_target_grid_freqbin.shape,exp_whole_original_grid.shape)

    delays_fir=np.zeros((1200,1))
    power_fir=np.zeros((1200,1))
    k=fft(l.rcv_msr_ir,axis=-1)[:,:(l.rcv_msr_ir.shape[1]//2)]
    diff=k[:,30]-exp_whole_original_grid[:,30]
    s.append(diff)

    #for i,k in enumerate(degree):
    #    plt.plot(np.arange(480),s[i],"--",label="Order "+str(k))
    #plt.xlabel("Measurement Nodes")
    #plt.legend()
    #plt.title("Relative Error For All Sampling Nodes , Frequency Bin = 5500 Hz ")
    #plt.show()
    #plt.savefig("IRCAM.png")

    outlier_peak=[]
    
    #Calculate the delay in the IR using peak picking.
    for nodes_fibo_grid in range(l.stack_phi_theta_fibo.shape[0]):


        peak=np.argmax(abs(np.real(ifft(exp_whole_target_grid[nodes_fibo_grid,:]))))
        #peaks,_=find_peaks(np.real(ifft(exp_whole_target_grid[nodes_fibo_grid,:])))

        if peak >= 75:
            outlier_peak.append(nodes_fibo_grid)
            #plt.plot(np.arange(683),ifft(exp_whole_target_grid[nodes_fibo_grid,:]))
            #plt.show()
        power_fir[nodes_fibo_grid,0]=20*np.log10(LA.norm(ifft(exp_whole_target_grid[nodes_fibo_grid,:])))
        delays_fir[nodes_fibo_grid, 0]=peak

        #print(peak)
        #print()
    
    '''
    #plt.clf()
    #print(outlier_peak)
    #l.plot(exp_original_grid_freqbin, exp_whole_target_grid[:, 100], freq_bin)
    #print("Interpolated point ",l.nn_kd_tree_original_grid.query([l.phi_fibo[200],l.theta_fibo[200]]))
    #print(peaks)
    #print(_)

    #axs[0].plot(np.arange(86),np.real(ifft(exp_whole_target_grid[200,:])),label="On Fibonnaci Interpolated Grid")
    #axs[0].plot(np.arange(86),np.real(ifft(exp_whole_original_grid[176,:])),label="Spherical Coeffs Expanded On Original Grid ")
    #axs[0].plot(np.arange(86),l.rcv_msr_ir[176,:],label="Original FIR Measurement Downsampled to 16khz ")
    #axs[0].legend()
    #axs[0].set_title("Matched Impulse Responses On An Approximate Point On Fibo Grid")

    #axs[i,0].legend()
    #axs[i,0].legend()

    #k=axs[1].scatter(np.rad2deg(l.phi_fibo),np.rad2deg(l.theta_fibo),c=delays_fir)
    #o=plt.colorbar(k, ax=axs[1])
    #axs[1].set_title("Measured Peak On Expanded Fibo Grid")
    #o.set_label("Samples")

    #s = axs[2].scatter(np.rad2deg(l.phi_fibo), np.rad2deg(l.theta_fibo), c=power_fir)
    #o=plt.colorbar(s, ax=axs[2])
    #axs[2].set_title("Observered DIrectivity Pattern On Expanded Fibo Grid")
    #o.set_label("dB")
    #plt.show()


#fig.tight_layout(pad=5.0)
#plt.show()


#plt.savefig("IRCAM.png")

# #exp_target_grid_freqbin


'''
    #l.neareast_neighbour(60,70,100)
    #To calculate Phase delay , ifft
    plt.clf()
    s=np.array(ifft(exp_whole_original_grid[50,:]),dtype=np.complex_)


    expanded_signal_fft_unphase=np.unwrap(np.angle(fft(s)[:l.samples_size_ir//2]))
    original_signal_fft_unphase = np.unwrap(np.angle(fft(l.rcv_msr_ir[50,:])[:l.samples_size_ir // 2]))



    plt.plot(fftfreq(l.samples_size_ir,d=1/44100)[1:l.samples_size_ir//2],-(expanded_signal_fft_unphase[1:]/fftfreq(l.samples_size_ir,d=1/44100)[1:l.samples_size_ir//2]),label="DSHT Expanded Signal")
    plt.plot(fftfreq(l.samples_size_ir, d=1 / 44100)[1:l.samples_size_ir // 2], -(original_signal_fft_unphase[1:] /fftfreq(l.samples_size_ir, d=1 / 44100)[1:l.samples_size_ir // 2] ),label="Original Signal")
    plt.ylabel("Radians/hz")
    plt.xlabel("Frequency In Hz ")
    plt.legend()
    plt.savefig("Phase_delay_50_node_receiver.png")

    plt.clf()
    plt.plot(np.arange(l.samples_size_ir),s.real,label="DSHT Expanded Signal")
    plt.plot(np.arange(l.samples_size_ir),l.rcv_msr_ir[50,:].real,color="orange",label="Original Signal")
    plt.legend()
    plt.savefig("inverse_fft_expanded_source_directivity_50_node.png")
    


    #To calculate relative error.


    #c=[np.linalg.norm(exp_whole_original_grid[:,freq]-l.freq_angles_fft[:,freq])/np.linalg.norm(l.freq_angles_fft[:,freq]) for freq in range(l.freq_angles_fft.shape[1])]

    #ll.append(c)
    

'''

#Some Notes
# higher degree means low phase delay error between the actual signal and the expanded signal
# Higher frequencies are better interpolated in the higher degree ,but that's not the case with the lower frequencies.


'''
print(np.array(ll).shape)

plt.clf()
for i in range(len(degree)):
    plt.plot(fftfreq(l.samples_size_ir,d=1/44100)[:l.samples_size_ir//2],np.array(ll[i]),c=colors[i],label=str(degree[i]))
plt.ylabel("Relative Error")
plt.xlabel("Frequency in Hz")
plt.legend()
plt.title("Relative error for frequency "+str(fftfreq(l.samples_size_ir,d=1/44100)[freq_bin])+"Hz for many degrees")
plt.savefig("Relative_error_func_freq_.png")
'''