import matplotlib
matplotlib.use('Agg')
import numpy as np
import sofa
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
src_sofa=sofa.Database.open("Soundfield_ST450_CUBE.sofa")
src_sofa.Metadata.dump()
print("Dimension")
src_sofa.Dimensions.dump()
print("Variables")
src_sofa.Variables.dump()


def plot_coordinates(coords, title):
    x0 = coords
    n0 = coords
    print(np.max(coords))
    print(np.min(coords))
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    #x, y, z = np.meshgrid(np.arange(0.0, 1.5, 0.003125),
                      #np.arange(0.0, 1.5, 0.003125),
                      #np.arange(0.0, 1.5, 0.003125))

    #print(x.shape)
    print(coords[:,0].shape)
    print(coords[:,1].shape)

    #ax.quiver(x, y, z, coords[:,0], coords[:,1], coords[:,2], length=0.1, normalize=True)
    ax.plot_trisurf(coords[:,0], coords[:,1], coords[:,2])
    #q = ax.quiver(x0[:, 0], x0[:, 1], x0[:, 2], n0[:, 0],
    #              n0[:, 1], n0[:, 2], length=0.1)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.ylabel('z (m)')
    plt.title(title)
    plt.savefig("src-pos.jpeg")
    #return q


print("Coordinate Dimension Size: C",src_sofa.Dimensions.C)
print("Scalar Dimension Size : I",src_sofa.Dimensions.I)
print("Number Of measurements : M",src_sofa.Dimensions.M)
print("Number Of Receivers : R",src_sofa.Dimensions.R)
print("Number Of emitters : E",src_sofa.Dimensions.E)
print("Number Of Data samples per measurements : N",src_sofa.Dimensions.N)
print("Largest data string size : S",src_sofa.Dimensions.S)



source_positions = src_sofa.Source.Position.get_values(system="cartesian")
plot_coordinates(source_positions,"Src Pos")
print(src_sofa.Data.IR.get_values().shape)
print(src_sofa.Data.SamplingRate.get_values()[0])
print(src_sofa.Emitter.Position.get_values().shape)
print(src_sofa.Source.Position.get_values().shape)
print(src_sofa.Receiver.Position.get_values().shape)
print(src_sofa.Listener.Position.get_values().shape)


measurement = 5
emitter = 0
legend = []

t = np.arange(0,src_sofa.Dimensions.N)*src_sofa.Data.SamplingRate.get_values(indices={"M":measurement})

plt.figure(figsize=(15, 5))
for receiver in np.arange(src_sofa.Dimensions.R):
    plt.plot(t, src_sofa.Data.IR.get_values(indices={"M":measurement, "R":receiver, "E":emitter}))
    legend.append('Receiver {0}'.format(receiver))
plt.title('HRIR at M={0} for emitter {1}'.format(measurement, emitter))
plt.legend(legend)
plt.xlabel('$t$ in s')
plt.ylabel(r'$h(t)$')
plt.grid()
plt.savefig("src-ir.jpeg")
