import plotly
import numpy as np
import pydicom
import os
import scipy.ndimage
from skimage import measure
from plotly.offline import plot
from plotly.tools import FigureFactory as FF
from sklearn.cluster import KMeans
from skimage import morphology
import matplotlib.pyplot as plt


class SomeClass:
    def __init__(self, path):
        self.path = path
        self.slices = None
        self.slice_thickness = None
        self.images = None
        self.verts = None
        self.faces = None
        self.norm = None
        self.val = None
        self.div = None


        print('loading images')
        self.load_scan()
        print('pre-processing images')
        self.get_pixels_hu()
        print('making a mesh')
        self.make_mesh()
        print('plotting a 3D model')
        self.plotly_3d()
  

    def load_scan(self):
        self.slices = [pydicom.read_file(self.path + '/' + s) for s in os.listdir(self.path)]
        self.slices.sort(key = lambda x: int(x.InstanceNumber))
        try:
            self.slice_thickness = np.abs(self.slices[0].ImagePositionPatient[2] - self.slices[1].ImagePositionPatient[2])
        except:
            self.slice_thickness = np.abs(self.slices[0].SliceLocation - self.slices[1].SliceLocation)
            
        for s in self.slices:
            s.SliceThickness = self.slice_thickness
  

    def get_pixels_hu(self):
        image = np.stack([s.pixel_array for s in self.slices])
        
        image = image.astype(np.int16)

        image[image == -2000] = 0
        
        intercept = self.slices[0].RescaleIntercept
        slope = self.slices[0].RescaleSlope
        
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)
            
        image += np.int16(intercept)
        
        self.images = np.array(image, dtype=np.int16)

        new_spacing = [1, 1, 1]

        spacing = map(float, ([self.slices[0].SliceThickness] + list(self.slices[0].PixelSpacing)))
        spacing = np.array(list(spacing))

        resize_factor = spacing / new_spacing
        new_real_shape = self.images.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / self.images.shape
        new_spacing = spacing / real_resize_factor
        
        self.images = scipy.ndimage.zoom(self.images, real_resize_factor)


    def make_mesh(self, threshold=100, step_size=2):
        p = self.images.transpose(2,1,0)

        self.verts, self.faces, self.norm, self.val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True)
        return self.verts, self.faces
  

    def plotly_3d(self):
        x, y, z = zip(*self.verts)

        colormap = ['rgb(236, 236, 212)', 'rgb(236, 236, 212)']

        fig = FF.create_trisurf(
            x=x,
            y=y,
            z=z,
            plot_edges=False,
            colormap=colormap,
            simplices=self.faces,
            backgroundcolor='rgb(64, 64, 64)',
            title="Interactive Visualization"
        )

        self.div = plot(fig, include_plotlyjs=False, output_type='div')
    

def make_lungmask(img, display=False):
    row_size= img.shape[0]
    col_size = img.shape[1]
    
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img,np.ones([3,3]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))

    labels = measure.label(dilation) # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0

    #
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    #
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation

    if (display):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask*img, cmap='gray')
        ax[2, 1].axis('off')
        
        plt.show()
    return mask*img


from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import shutil

UPLOAD_FOLDER = 'uploads'
app = Flask(__name__, template_folder='templates/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 900*1000*1000


@app.route("/")
def home():
    folder = 'uploads'
    if os.path.isdir(os.path.join(os.getcwd(), folder)):
        shutil.rmtree(os.path.join(os.getcwd(), folder))
        os.mkdir(os.path.join(os.getcwd(), folder))
    path = os.getcwd()
    UPLOAD_FOLDER = os.path.join(path, 'uploads')
    return render_template('file1.html')

@app.route("/file", methods=['POST'])
def file():
    files = request.files.getlist('file[]')
    for f in files:
        fname = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))


    obj = SomeClass(UPLOAD_FOLDER)
    
    obj.div = obj.div.replace('height:800px; width:800px;', 'width:50%; margin: 0 auto')


    img = obj.images[260]
    make_lungmask(img, display=True)
    
        
    
    return render_template('test.html', data=obj.div)


app.run()