import os
import shutil
import argparse

import time
import datetime

import math
import random
import string

import numpy
import cv2
import qrcode
from PIL import Image, ImageFont, ImageDraw
from pascal_voc_writer import Writer

LABEL_SHAPE = (640,480)
QR_SIZE = 6
QR_PIXEL_WIDTH = 25
QR_ORIGIN = (250,75)
RECT_ORIGIN = (75,250)
RECT_WIDTH = 490
RECT_HEIGHT = 120
LOGO_ORIGIN = (75,75)
LOGO_WIDTH = 150.0
SCALE_FACTOR = 0.7
INK_BLOTCH_MIN = 10
INK_BLOTCH_MAX = 20

def rand(val):
    return int(numpy.random.random() * val)

class GenerateLabel:
    def __init__(self):
        self.img=numpy.array(Image.new("RGB", LABEL_SHAPE,(255,255,255)))
        
    def getRandomStr(self, size):
        randStr = ''.join(random.choice(string.ascii_letters + '      ' + string.digits) for _ in range(size))
        return randStr
    
    def euler_to_mat(self, yaw, pitch, roll):
        # Rotate clockwise about the Y-axis
        c, s = math.cos(yaw), math.sin(yaw)
        M = numpy.matrix([[  c, 0.,  s], [ 0., 1., 0.], [ -s, 0.,  c]])

        # Rotate clockwise about the X-axis
        c, s = math.cos(pitch), math.sin(pitch)
        M = numpy.matrix([[ 1., 0., 0.], [ 0.,  c, -s], [ 0.,  s,  c]]) * M

        # Rotate clockwise about the Z-axis
        c, s = math.cos(roll), math.sin(roll)
        M = numpy.matrix([[  c, -s, 0.], [  s,  c, 0.], [ 0., 0., 1.]]) * M

        return M
    
    def make_affine_transform(self, from_shape, to_shape, 
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
        out_of_bounds = 0

        from_size = numpy.array([[from_shape[1], from_shape[0]]]).T
        to_size = numpy.array([[to_shape[1], to_shape[0]]]).T

        scale = random.uniform((min_scale + max_scale) * SCALE_FACTOR -
                               (max_scale - min_scale) * SCALE_FACTOR * scale_variation,
                               (min_scale + max_scale) * SCALE_FACTOR +
                               (max_scale - min_scale) * SCALE_FACTOR * scale_variation)
        #if scale > max_scale or scale < min_scale:
            #out_of_bounds = 1
        if scale < min_scale:
            out_of_bounds = 1
        roll = random.uniform(-1.0, 1.0) * rotation_variation
        pitch = random.uniform(-0.15, 0.15) * rotation_variation
        yaw = random.uniform(-0.15, 0.15) * rotation_variation

        # Compute a bounding box on the skewed input image (`from_shape`).
        M = self.euler_to_mat(yaw, pitch, roll)[:2, :2]
        h = from_shape[0]
        w = from_shape[1]
        corners = numpy.matrix([[-w, +w, -w, +w],
                                [-h, -h, +h, +h]]) * 0.5
        skewed_size = numpy.array(numpy.max(M * corners, axis=1) -
                                  numpy.min(M * corners, axis=1))

        # Set the scale as large as possible such that the skewed and scaled shape
        # is less than or equal to the desired ratio in either dimension.
        scale *= numpy.min(to_size / skewed_size)

        # Set the translation such that the skewed and scaled image falls within
        # the output shape's bounds.
        trans = (numpy.random.random((2,1)) - 0.5) * translation_variation
        trans = ((2.0 * trans) ** 5.0) / 2.0
        if numpy.any(trans < -0.5) or numpy.any(trans > 0.5):
            out_of_bounds = 1
        trans = (to_size - skewed_size * scale) * trans

        center_to = to_size / 2.
        center_from = from_size / 2.

        M = self.euler_to_mat(yaw, pitch, roll)[:2, :2]
        M *= scale
        M = numpy.hstack([M, trans + center_to - M * center_from])

        return M, out_of_bounds
    
    def addRoundedRectangleBorder(self, img):
        height, width, channels = img.shape

        border_radius = int(30)
        line_thickness = int(1)
        edge_shift = int(line_thickness/2.0)
        color = (0, 0, 0)

        #draw lines
        #top
        cv2.line(img, (border_radius, edge_shift), (width - border_radius, edge_shift), color, line_thickness)
        #bottom
        cv2.line(img, (border_radius, height-line_thickness), (width - border_radius, height-line_thickness), color, line_thickness)
        #left
        cv2.line(img, (edge_shift, border_radius), (edge_shift, height  - border_radius), color, line_thickness)
        #right
        cv2.line(img, (width - line_thickness, border_radius), (width - line_thickness, height  - border_radius), color, line_thickness)

        #corners
        cv2.ellipse(img, (border_radius+ edge_shift, border_radius+edge_shift), (border_radius, border_radius), 180, 0, 90, color, line_thickness)
        cv2.ellipse(img, (width-(border_radius+line_thickness), border_radius), (border_radius, border_radius), 270, 0, 90, color, line_thickness)
        cv2.ellipse(img, (width-(border_radius+line_thickness), height-(border_radius + line_thickness)), (border_radius, border_radius), 10, 0, 90, color, line_thickness)
        cv2.ellipse(img, (border_radius+edge_shift, height-(border_radius + line_thickness)), (border_radius, border_radius), 90, 0, 90, color, line_thickness)

        return img
    
    def createMask(self, shape, radius):
        out = numpy.ones(shape)
        out[:radius, :radius] = 0.
        out[-radius:, :radius] = 0.
        out[:radius, -radius:] = 0.
        out[-radius:, -radius:] = 0.

        cv2.circle(out, (radius, radius), radius, (1.,1.,1.), -1)
        cv2.circle(out, (radius, shape[0] - radius), radius, (1.,1.,1.), -1)
        cv2.circle(out, (shape[1] - radius, radius), radius, (1.,1.,1.), -1)
        cv2.circle(out, (shape[1] - radius, shape[0] - radius), radius, (1.,1.,1.), -1)
        return out
    
    
    def addLogo(self, img, logoPath):
        logo = Image.open(logoPath)
        scale = float(logo.size[0]/LOGO_WIDTH)
        new_width = int(logo.size[0]/scale)
        new_height = int(logo.size[1]/scale)
        self.logo_height = new_height
        logo = logo.resize((new_width, new_height))
        pil_img = Image.fromarray(img)
        pil_img.paste(logo, LOGO_ORIGIN)
        pasted = numpy.array(pil_img)
        return pasted
    
    def addRandomQRCode(self, img):
        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=QR_SIZE, border=0)
        qr.add_data(self.getRandomStr(24))
        qr.make(fit=True)
        gen_qr = qr.make_image(fill_color='black', back_color='white')
        
        pil_img = Image.fromarray(img)
        pil_img.paste(gen_qr, QR_ORIGIN)
        pasted = numpy.array(pil_img)
        return pasted
    
    def addSampleRandomQRCode(self, img):
        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=QR_SIZE, border=0)
        qr.add_data('https://www.slalom.com')
        qr.make(fit=True)
        gen_qr = qr.make_image(fill_color='black', back_color='white')
        
        pil_img = Image.fromarray(img)
        pil_img.paste(gen_qr, QR_ORIGIN)
        pasted = numpy.array(pil_img)
        return pasted
    
    def addRectangularTextLabel(self, img):
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle((RECT_ORIGIN, (RECT_ORIGIN[0]+RECT_WIDTH,RECT_ORIGIN[1]+RECT_HEIGHT)), fill='black')
        font = ImageFont.truetype('/Library/Fonts/Microsoft Sans Serif.ttf', 20)
        draw.text((RECT_ORIGIN[0]+25,RECT_ORIGIN[1]+20), '100 Pine Street, Suite 2500', font=font)
        draw.text((RECT_ORIGIN[0]+25,RECT_ORIGIN[1]+50), 'San Francisco, CA 94111', font=font)
        draw.text((RECT_ORIGIN[0]+25,RECT_ORIGIN[1]+80), 'RANDOM: '+self.getRandomStr(24), font=font)
        npimg = numpy.array(pil_img)
        return npimg
    
    def addSampleRectangularTextLabel(self, img):
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle((RECT_ORIGIN, (RECT_ORIGIN[0]+RECT_WIDTH,RECT_ORIGIN[1]+RECT_HEIGHT)), fill='black')
        font = ImageFont.truetype('/Library/Fonts/Microsoft Sans Serif.ttf', 40)
        draw.text((RECT_ORIGIN[0]+25,RECT_ORIGIN[1]+20), 'WARNING: This is a test', font=font)
        npimg = numpy.array(pil_img)
        return npimg
    
    def addInkBlotch(self, img):        
        pil_img = Image.fromarray(img)
        npimg = numpy.array(pil_img)
        maxX = 530
        minX = 0
        result_array = []
        minLineNum = INK_BLOTCH_MIN
        maxLineNum = INK_BLOTCH_MAX
        for x in range(random.randint(minLineNum, maxLineNum)):
            tempX = random.randint(minX, maxX)
            tempY = random.randint(minX, maxX)
            result_array.append([tempX, tempY])
            # print result_array
        pts = numpy.array(result_array, numpy.int32)
        r = lambda: random.randint(0,255)
        random_color = (r(),r(),r())
        cv2.polylines(npimg,[pts],False,random_color, random.randint(2,8))
        return npimg

    
    def addCaution(self,img):
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        #draw.rectangle(((75,430), (75+RECT_WIDTH, 450)), fill=(255,255,0))
        font = ImageFont.truetype('/Library/Fonts/Arial Italic.ttf', 17)
        draw.text((75,400), 'CAUTION: This label is generated for Slalom internal testing only', font=font, fill=(255,0,0,255))
        npimg = numpy.array(pil_img)
        return npimg
    
    def addGauss(self, img, level):
        return cv2.blur(img, (level * 2 + 1, level * 2 + 1))
    
    def addNoiseSingleChannel(self, single):
        diff = 255 - single.max();
        noise = numpy.random.normal(0, 1+rand(100), single.shape);
        noise = (noise - noise.min())/(noise.max()-noise.min())
        noise= diff*noise;
        noise= noise.astype(numpy.uint8)
        dst = single + noise
        return dst
    
    def addNoise(self, img):
        img[:,:,0] = self.addNoiseSingleChannel(img[:,:,0]);
        img[:,:,1] = self.addNoiseSingleChannel(img[:,:,1]);
        img[:,:,2] = self.addNoiseSingleChannel(img[:,:,2]);
        return img;
    
    def tfactor(self,img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV);

        hsv[:,:,0] = hsv[:,:,0]*(0.8+ numpy.random.random()*0.2);
        hsv[:,:,1] = hsv[:,:,1]*(0.4+ numpy.random.random()*0.6);
        hsv[:,:,2] = hsv[:,:,2]*(0.4+ numpy.random.random()*0.6);

        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR);
        return img
    
    def generate_bg(self, bkg_dir):
        found = False
        while not found:
            fname = bkg_dir + "/{:08d}.jpg".format(random.randint(0, len(os.listdir(bkg_dir)) - 2))
            print('selected {} as background'.format(fname))
            bg = cv2.imread(fname, 1)
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
            
            #random rotation
            rotate_M = cv2.getRotationMatrix2D((bg.shape[1]/2,bg.shape[0]/2),random.randint(0,3) * 90,1)
            bg = cv2.warpAffine(bg,rotate_M,(bg.shape[1],bg.shape[0]))
            
            if (bg.shape[1] >= LABEL_SHAPE[0] and
                bg.shape[0] >= LABEL_SHAPE[1]):
                found = True
        return bg
    

    def genRealSamples(self, batchSize, outputPath, logoPath):
        if not os.path.exists(outputPath+'/Samples'):
            os.makedirs(outputPath+'/Samples')
            
        for i in xrange(batchSize):
            genSample = self.img

            #genSample = self.addLogo(genSample, logoPath)
            genSample = self.addRoundedRectangleBorder(genSample)
            genSample = self.addSampleRandomQRCode(genSample)
            genSample = self.addSampleRectangularTextLabel(genSample)
            genSample = self.addCaution(genSample)
            #genSample = self.tfactor(genSample)

            out = genSample
            
            sample_filename = os.path.join(outputPath+'/Samples', '9' + str(i).zfill(5) + '.jpg')
            pil_image = Image.fromarray(out.astype('uint8'))
            pil_image.save(sample_filename, format='JPEG', subsampling=0, quality=100)
            
            print('{} sample generated.'.format(sample_filename))
        
        
    def genBatch(self, batchSize, outputPath, logoPath, backgroundDir, classNames, debugFlag):
        
        shutil.rmtree(outputPath)
        
        class_names = classNames.split(",")
        
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
            
        if not os.path.exists(outputPath+'/JPEGImages'):
            os.makedirs(outputPath+'/JPEGImages')
            
        if not os.path.exists(outputPath+'/Annotations'):
            os.makedirs(outputPath+'/Annotations')
            
        if not os.path.exists(outputPath+'/ImageSets'):
            os.makedirs(outputPath+'/ImageSets')
            
        if not os.path.exists(outputPath+'/ImageSets/Main'):
            os.makedirs(outputPath+'/ImageSets/Main')
            
        main_val_file = open(outputPath+'/ImageSets/Main/val.txt','w')
        main_train_file = open(outputPath+'/ImageSets/Main/train.txt','w')
        main_trainval_file = open(outputPath+'/ImageSets/Main/trainval.txt','w')
        
        gen_log_file = open(outputPath+'/gen.log','w')
            
        for class_name_idx, class_name in enumerate(class_names):
            for i in range(batchSize):
                self.bkg = self.generate_bg(backgroundDir)
                generatedLabel = self.img

                generatedLabel = self.addLogo(generatedLabel, logoPath)
                generatedLabel = self.addRoundedRectangleBorder(generatedLabel)
                generatedLabel = self.addRandomQRCode(generatedLabel)
                generatedLabel = self.addRectangularTextLabel(generatedLabel)
                generatedLabel = self.addCaution(generatedLabel)
                
                if class_name == 'anomalous_label':
                    generatedLabel = self.addInkBlotch(generatedLabel)

                labelMask = self.createMask(generatedLabel.shape, 30)

                generatedBackground = self.bkg

                M, out_of_bounds = self.make_affine_transform(
                                    from_shape=generatedLabel.shape,
                                    to_shape=generatedBackground.shape,
                                    min_scale=0.5,
                                    max_scale=0.8,
                                    rotation_variation=1.0,
                                    scale_variation=1.5,
                                    translation_variation=1.02)

                label_topleft = tuple(M.dot(numpy.array((0,0) + (1,))).tolist()[0])
                label_topright = tuple(M.dot(numpy.array((LABEL_SHAPE[0],0) + (1,))).tolist()[0])
                label_bottomleft = tuple(M.dot(numpy.array((0,LABEL_SHAPE[1]) + (1,))).tolist()[0])
                label_bottomright = tuple(M.dot(numpy.array((LABEL_SHAPE[0],LABEL_SHAPE[1]) + (1,))).tolist()[0])

                label_tups = (label_topleft, label_topright, label_bottomleft, label_bottomright)
                label_xmin = int(min(label_tups, key=lambda item:item[0])[0])
                label_xmax = int(max(label_tups, key=lambda item:item[0])[0])
                label_ymin = int(min(label_tups, key=lambda item:item[1])[1])
                label_ymax = int(max(label_tups, key=lambda item:item[1])[1])

                qr_topleft = tuple(M.dot(numpy.array((QR_ORIGIN[0],QR_ORIGIN[1]) + (1,))).tolist()[0])
                qr_topright = tuple(M.dot(numpy.array((QR_ORIGIN[0]+QR_SIZE*QR_PIXEL_WIDTH,QR_ORIGIN[1]) + (1,))).tolist()[0])
                qr_bottomleft = tuple(M.dot(numpy.array((QR_ORIGIN[0],QR_ORIGIN[1]+QR_SIZE*QR_PIXEL_WIDTH) + (1,))).tolist()[0])
                qr_bottomright = tuple(M.dot(numpy.array((QR_ORIGIN[0]+QR_SIZE*QR_PIXEL_WIDTH,QR_ORIGIN[1]+QR_SIZE*QR_PIXEL_WIDTH) + (1,))).tolist()[0])

                qr_tups = (qr_topleft, qr_topright, qr_bottomleft, qr_bottomright)
                qr_xmin = int(min(qr_tups, key=lambda item:item[0])[0])
                qr_xmax = int(max(qr_tups, key=lambda item:item[0])[0])
                qr_ymin = int(min(qr_tups, key=lambda item:item[1])[1])
                qr_ymax = int(max(qr_tups, key=lambda item:item[1])[1])

                rect_topleft = tuple(M.dot(numpy.array((RECT_ORIGIN[0],RECT_ORIGIN[1]) + (1,))).tolist()[0])
                rect_topright = tuple(M.dot(numpy.array((RECT_ORIGIN[0]+RECT_WIDTH,RECT_ORIGIN[1]) + (1,))).tolist()[0])
                rect_bottomleft = tuple(M.dot(numpy.array((RECT_ORIGIN[0],RECT_ORIGIN[1]+RECT_HEIGHT) + (1,))).tolist()[0])
                rect_bottomright = tuple(M.dot(numpy.array((RECT_ORIGIN[0]+RECT_WIDTH,RECT_ORIGIN[1]+RECT_HEIGHT) + (1,))).tolist()[0])

                rect_tups = (rect_topleft, rect_topright, rect_bottomleft, rect_bottomright)
                rect_xmin = int(min(rect_tups, key=lambda item:item[0])[0])
                rect_xmax = int(max(rect_tups, key=lambda item:item[0])[0])
                rect_ymin = int(min(rect_tups, key=lambda item:item[1])[1])
                rect_ymax = int(max(rect_tups, key=lambda item:item[1])[1])

                logo_topleft = tuple(M.dot(numpy.array((LOGO_ORIGIN[0],LOGO_ORIGIN[1]) + (1,))).tolist()[0])
                logo_topright = tuple(M.dot(numpy.array((LOGO_ORIGIN[0]+LOGO_WIDTH,LOGO_ORIGIN[1]) + (1,))).tolist()[0])
                logo_bottomleft = tuple(M.dot(numpy.array((LOGO_ORIGIN[0],LOGO_ORIGIN[1]+self.logo_height) + (1,))).tolist()[0])
                logo_bottomright = tuple(M.dot(numpy.array((LOGO_ORIGIN[0]+LOGO_WIDTH,LOGO_ORIGIN[1]+self.logo_height) + (1,))).tolist()[0])

                logo_tups = (logo_topleft, logo_topright, logo_bottomleft, logo_bottomright)
                logo_xmin = int(min(logo_tups, key=lambda item:item[0])[0])
                logo_xmax = int(max(logo_tups, key=lambda item:item[0])[0])
                logo_ymin = int(min(logo_tups, key=lambda item:item[1])[1])
                logo_ymax = int(max(logo_tups, key=lambda item:item[1])[1])

                generatedLabel = cv2.warpAffine(generatedLabel, M, (generatedBackground.shape[1], generatedBackground.shape[0]))
                labelMask = cv2.warpAffine(labelMask, M, (generatedBackground.shape[1], generatedBackground.shape[0]))

                #light condition
                generatedLabel = self.tfactor(generatedLabel)

                out = generatedLabel * labelMask + generatedBackground * (1.0 - labelMask)

                ###debug
                if (debugFlag):
                    cv2.line(out, (int(label_topleft[0]), int(label_topleft[1])), (int(label_topright[0]),int(label_topright[1])), (0,255,0), 2)
                    cv2.line(out, (int(label_topright[0]), int(label_topright[1])), (int(label_bottomright[0]),int(label_bottomright[1])), (0,255,0), 2)
                    cv2.line(out, (int(label_bottomright[0]), int(label_bottomright[1])), (int(label_bottomleft[0]),int(label_bottomleft[1])), (0,255,0), 2)
                    cv2.line(out, (int(label_bottomleft[0]), int(label_bottomleft[1])), (int(label_topleft[0]),int(label_topleft[1])), (0,255,0), 2)
                    cv2.rectangle(out, (label_xmin, label_ymin), (label_xmax, label_ymax), (255,0,0), 1)

                    cv2.line(out, (int(qr_topleft[0]), int(qr_topleft[1])), (int(qr_topright[0]),int(qr_topright[1])), (0,0,255), 1)
                    cv2.line(out, (int(qr_topright[0]), int(qr_topright[1])), (int(qr_bottomright[0]),int(qr_bottomright[1])), (0,0,255), 1)
                    cv2.line(out, (int(qr_bottomright[0]), int(qr_bottomright[1])), (int(qr_bottomleft[0]),int(qr_bottomleft[1])), (0,0,255), 1)
                    cv2.line(out, (int(qr_bottomleft[0]), int(qr_bottomleft[1])), (int(qr_topleft[0]),int(qr_topleft[1])), (0,0,255), 1)
                    cv2.rectangle(out, (qr_xmin, qr_ymin), (qr_xmax, qr_ymax), (255,0,0), 1)

                    cv2.line(out, (int(rect_topleft[0]), int(rect_topleft[1])), (int(rect_topright[0]),int(rect_topright[1])), (0,255,255), 1)
                    cv2.line(out, (int(rect_topright[0]), int(rect_topright[1])), (int(rect_bottomright[0]),int(rect_bottomright[1])), (0,255,255), 1)
                    cv2.line(out, (int(rect_bottomright[0]), int(rect_bottomright[1])), (int(rect_bottomleft[0]),int(rect_bottomleft[1])), (0,255,255), 1)
                    cv2.line(out, (int(rect_bottomleft[0]), int(rect_bottomleft[1])), (int(rect_topleft[0]),int(rect_topleft[1])), (0,255,255), 1)
                    cv2.rectangle(out, (rect_xmin, rect_ymin), (rect_xmax, rect_ymax), (255,0,0), 1)

                    cv2.line(out, (int(logo_topleft[0]), int(logo_topleft[1])), (int(logo_topright[0]),int(logo_topright[1])), (16,255,224), 1)
                    cv2.line(out, (int(logo_topright[0]), int(logo_topright[1])), (int(logo_bottomright[0]),int(logo_bottomright[1])), (16,255,224), 1)
                    cv2.line(out, (int(logo_bottomright[0]), int(logo_bottomright[1])), (int(logo_bottomleft[0]),int(logo_bottomleft[1])), (16,255,224), 1)
                    cv2.line(out, (int(logo_bottomleft[0]), int(logo_bottomleft[1])), (int(logo_topleft[0]),int(logo_topleft[1])), (16,255,224), 1)
                    cv2.rectangle(out, (logo_xmin, logo_ymin), (logo_xmax, logo_ymax), (255,0,0), 1)


                # if needed, resize here
                # gauss
                out = self.addGauss(out, 0+rand(2))
                # minor noise
                out += numpy.random.normal(scale=0.2, size=out.shape)
                #out = self.addNoise(out)
                
                initial_val = '1'
                total_index = (class_name_idx * batchSize) + i

                img_filename = os.path.join(outputPath+'/JPEGImages', initial_val + str(total_index).zfill(5) + '.jpg')
                xml_filename = os.path.join(outputPath+'/Annotations', initial_val + str(total_index).zfill(5) + '.xml')
                
                pil_image = Image.fromarray(out.astype('uint8'))
                pil_image.save(img_filename, format='JPEG', subsampling=0, quality=100)
                
                annotator = Writer(img_filename, pil_image.size[0], pil_image.size[1])
                
                annotator.addObject(class_name,label_xmin,label_ymin,label_xmax,label_ymax,'Unspecified',out_of_bounds)
                annotator.addObject('qrcode',qr_xmin,qr_ymin,qr_xmax,qr_ymax)
                annotator.addObject('title',rect_xmin,rect_ymin,rect_xmax,rect_ymax)
                annotator.addObject('logo',logo_xmin,logo_ymin,logo_xmax,logo_ymax)
                annotator.save(xml_filename)
                
                if i % (batchSize / 10) == 0:
                    unformatted_ts = datetime.datetime.fromtimestamp(time.time())
                    ts = unformatted_ts.strftime('%Y-%m-%d %H:%M:%S')
                    log_debug_string = '### {} ### Generated Files: {}, {}\n'.format(ts, img_filename, xml_filename)
                    gen_log_file.write(log_debug_string)
                    print(log_debug_string)


                main_trainval_file.write(initial_val + str(total_index).zfill(5) + '\n')
                
                is_train_id = (i < batchSize * 0.8)
                if is_train_id:
                    main_train_file.write(initial_val + str(total_index).zfill(5) + '\n')
                else:
                    main_val_file.write(initial_val + str(total_index).zfill(5) + '\n')
                
                for class_name_file in class_names:
                    
                    label_val_file = open(outputPath+'/ImageSets/Main/' + class_name_file + '_val.txt','a')
                    label_train_file = open(outputPath+'/ImageSets/Main/' + class_name_file + '_train.txt','a')
                    label_trainval_file = open(outputPath+'/ImageSets/Main/' + class_name_file + '_trainval.txt','a')
                    
                    presence_val = ' -1\n'
                    
                    if class_name_file == class_name:
                        presence_val = ' 1\n'
                        
                    label_trainval_file.write(initial_val + str(total_index).zfill(5) + presence_val)
                    
                    if is_train_id:
                        label_train_file.write(initial_val + str(total_index).zfill(5) + presence_val)
                    else:
                        label_val_file.write(initial_val + str(total_index).zfill(5) + presence_val)
                        
                qrcode_val_file = open(outputPath+'/ImageSets/Main/qrcode_val.txt','a')
                qrcode_train_file = open(outputPath+'/ImageSets/Main/qrcode_train.txt','a')
                qrcode_trainval_file = open(outputPath+'/ImageSets/Main/qrcode_trainval.txt','a')
                title_val_file = open(outputPath+'/ImageSets/Main/title_val.txt','a')
                title_train_file = open(outputPath+'/ImageSets/Main/title_train.txt','a')
                title_trainval_file = open(outputPath+'/ImageSets/Main/title_trainval.txt','a')
                logo_val_file = open(outputPath+'/ImageSets/Main/logo_val.txt','a')
                logo_train_file = open(outputPath+'/ImageSets/Main/logo_train.txt','a')
                logo_trainval_file = open(outputPath+'/ImageSets/Main/logo_trainval.txt','a')
                qrcode_trainval_file.write(initial_val + str(total_index).zfill(5) + ' 1\n')
                title_trainval_file.write(initial_val + str(total_index).zfill(5) + ' 1\n')
                logo_trainval_file.write(initial_val + str(total_index).zfill(5) + ' 1\n')
                
                if is_train_id:
                    qrcode_train_file.write(initial_val + str(total_index).zfill(5) + ' 1\n')
                    title_train_file.write(initial_val + str(total_index).zfill(5) + ' 1\n')
                    logo_train_file.write(initial_val + str(total_index).zfill(5) + ' 1\n')
                else:
                    qrcode_val_file.write(initial_val + str(total_index).zfill(5) + ' 1\n')
                    title_val_file.write(initial_val + str(total_index).zfill(5) + ' 1\n')
                    logo_val_file.write(initial_val + str(total_index).zfill(5) + ' 1\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backgrounds-dir', default='./backgrounds')
    parser.add_argument('--out-dir', default='./generated')
    parser.add_argument('--logo-path', default='./images/slalom_logo.jpg')
    parser.add_argument('--make-num', default=1000, type=int)
    parser.add_argument('--class-names', default='label')
    parser.add_argument('--samples', action='store_true')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()

def main(args):
    print('Generating...')
    G = GenerateLabel()
    if (args.samples):
        G.genRealSamples(args.make_num, args.out_dir, args.logo_path)
    else:
        G.genBatch(args.make_num, args.out_dir, args.logo_path, args.backgrounds_dir, args.class_names, args.debug)

if __name__ == '__main__':
    main(parse_args())