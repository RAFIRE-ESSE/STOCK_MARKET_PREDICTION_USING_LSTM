import yfinance,numpy
from moviepy.editor import *
from moviepy.config import change_settings
import cv2,numpy,os
import numpy as np
from PIL import Image,ImageFilter
import matplotlib. pyplot as plt
import matplotlib,pandas
from pandas import DataFrame
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import LSTM
from datetime import datetime, timedelta

#os.chmod(r"/etc/ImageMagick-6/magic.xml", 0777)
class demon_class:
    change_settings({"IMAGEMAGICK_BINARY": r"C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})
    def rect_with_rounded_corners(image, r, t, c):
        c += (255, )
        h, w = image.shape[:2]
        new_image = numpy.ones((h+2*t, w+2*t, 4), numpy.uint8) * 255
        new_image[:, :, 3] = 0
        new_image = cv2.ellipse(new_image, (int(r+t/2), int(r+t/2)), (r, r), 180, 0, 90, c, t)
        new_image = cv2.ellipse(new_image, (int(w-r+3*t/2-1), int(r+t/2)), (r, r), 270, 0, 90, c, t)
        new_image = cv2.ellipse(new_image, (int(r+t/2), int(h-r+3*t/2-1)), (r, r), 90, 0, 90, c, t)
        new_image = cv2.ellipse(new_image, (int(w-r+3*t/2-1), int(h-r+3*t/2-1)), (r, r), 0, 0, 90, c, t)
        new_image = cv2.line(new_image, (int(r+t/2), int(t/2)), (int(w-r+3*t/2-1), int(t/2)), c, t)
        new_image = cv2.line(new_image, (int(t/2), int(r+t/2)), (int(t/2), int(h-r+3*t/2)), c, t)
        new_image = cv2.line(new_image, (int(r+t/2), int(h+3*t/2)), (int(w-r+3*t/2-1), int(h+3*t/2)), c, t)
        new_image = cv2.line(new_image, (int(w+3*t/2), int(r+t/2)), (int(w+3*t/2), int(h-r+3*t/2)), c, t)
        mask = new_image[:, :, 3].copy()
        mask = cv2.floodFill(mask, None, (int(w/2+t), int(h/2+t)), 128)[1]
        mask[mask != 128] = 0
        mask[mask == 128] = 1
        mask = numpy.stack((mask, mask, mask), axis=2)
        temp = numpy.zeros_like(new_image[:, :, :3])
        temp[(t-1):(h+t-1), (t-1):(w+t-1)] = image.copy()
        new_image[:, :, :3] = new_image[:, :, :3] * (1 - mask) + temp * mask
        temp = new_image[:, :, 3].copy()
        new_image[:, :, 3] = cv2.floodFill(temp, None, (int(w/2+t), int(h/2+t)), 255)[1]
        return new_image
    def convert_frames_to_video(time,devil_image=r"static/devil_catch/angel.png",pathOut=r'static/devil_catch/video.mp4',fps=2.0):
        frame_array,img=[],cv2.imread(devil_image)
        files = [devil_image]*(time*int(fps))
        for i in range(len(files)):
            filename=files[i]
            height,width,layers =img.shape
            size=(width,height)
            frame_array.append(img)
        out=cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        for i in range(len(frame_array)):
            out.write(frame_array[i])
        out.release()
        del frame_array,files,filename,height,width,layers,size,out
    def hex_to_rgb(value):
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    def devil_ploter(devil_true_data,devil_pridict_data,cur):
        def devil_set_out(devil_obj):
            devil_obj.spines['top'].set_visible(False)
            devil_obj.spines['right'].set_visible(False)
            devil_obj.spines['left'].set_visible(False)
            devil_obj.spines['bottom'].set_visible(False)
            devil_obj.grid(color='grey', linestyle='-', linewidth=0.25, alpha=1)
            devil_obj.legend()
            
        div_,ang_=plt.subplots(figsize=(10, 5))
        ang_.plot(devil_true_data, label="TRUE Value")
        devil_set_out(ang_)
        plt.xlabel("Time Scale")
        plt.ylabel(f"Scaled {cur}")
        plt.savefig(r"static/devil_graph_out/devil_.jpg",dpi=100)

        div,ang=plt.subplots(figsize=(10, 5))
        ang.plot(devil_true_data[len(devil_true_data)-130:len(devil_true_data)],label="TRUE Value",linewidth=1.7)
        devil_pridict_data=numpy.insert(devil_pridict_data,0,devil_true_data[len(devil_true_data)-1])
        dates=pandas.date_range((datetime.now()+timedelta(0)).strftime('%d-%m-%Y'), periods=11)
        pridicter_angel=DataFrame(devil_pridict_data,index=dates)
        ang.plot(pridicter_angel,label="Prediction by LSTM",linewidth=1.7)
        devil_set_out(ang)
        plt.xlabel("Time Scale")
        plt.ylabel(f"Scaled {cur}")
        plt.savefig(r"static/devil_graph_out/devil.jpg",dpi=100)
        Image.open(r"static/devil_graph_out/devil.jpg").crop((0, 40, 1000, 500)).save(r"static/devil_graph_out/devil.jpg")
        Image.open(r"static/devil_graph_out/devil_.jpg").crop((0, 40, 1000, 500)).save(r"static/devil_graph_out/devil_.jpg")
        
class devil_class:
    def devil_pridicter(data,X_train=[],y_train=[]):
        for i in range(0,len(data),5):
            if (i+5)-1<len(data):
                X_train.append(data[i:i+4])
                y_train.append(data[(i+5)-1])
        X_train,y_train=np.array(X_train),np.array(y_train)
        trainX=np.array(X_train)
        print(trainX[len(trainX)-1],y_train[len(y_train)-1])
        X_train=trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
        lstm=Sequential()
        lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation="relu", return_sequences=False))
        lstm.add(Dense(1))
        lstm.compile(loss="mean_squared_error", optimizer="adam")
        lstm.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, shuffle=False)
        devil,angel=numpy.array([[data[len(data)-4:len(data)]]]),[]
        for i in range(10):
            y_pred= lstm.predict(devil)
            devil=numpy.delete(devil,0,2)
            devil=numpy.append(devil,[[[y_pred[0][0]]]],axis=2)
            angel.append(y_pred)
        #lstm.reset_state()
        print(numpy.array(angel).reshape(-1))
        return numpy.array(angel).reshape(-1)
        
    def devil_connect_out(tiker_devil):
        yfinance.pdr_override()
        evil_df,msft=yfinance.download(tiker_devil),yfinance.Ticker(tiker_devil)
        msft,cur=msft.info['longName'],msft.info['currency']
        pridiction=devil_class.devil_pridicter(numpy.array(evil_df['Close'][len(evil_df['Close'])%5::]))
        demon_class.devil_ploter(evil_df['Close'],pridiction,cur)
        return [[i,evil_df._get_value(evil_df.index[len(evil_df.index)-1],i)] for i in [j for j in evil_df]],evil_df['Close'][len(evil_df)-2],pridiction,msft,cur
        
    def devil_front_out(devil_list_out,angel_close,pridict,name_out,devil_out,cur):
        if len(name_out)>32:
            name_out=f"{name_out[0:25]}..."
        Image.fromarray(numpy.array(demon_class.hex_to_rgb("#ffffff")*(720*1080), dtype=numpy.uint8).reshape((1080,720,3)) , 'RGB').save(r"static/devil_catch/angel.png")
        demon_class.convert_frames_to_video(2)
        do_devil,sub_value=(devil_list_out[3][1]-angel_close),str(((devil_list_out[3][1]-angel_close)/devil_list_out[3][1])*100).split('.')
        clip,devil,count,count_,x,main_devil,angel_do=VideoFileClip(r"static/devil_catch/video.mp4"),[],930,0,30,str(devil_list_out[3][1]).split('.'),str(do_devil).split('.')
        devil.append(TextClip(f"{main_devil[0]}.{main_devil[1][0:2]} {cur}",font="Arial Black",fontsize = 720//15,color="gray").set_pos((30,30)))
        if do_devil>0:
            devil.append(TextClip(f"+{angel_do[0]}.{angel_do[1][0:2]} ({sub_value[0]}.{sub_value[1][0:2]}%)↑ today",font="Arial Black",fontsize = 720//35,color="gray").set_pos((30,90)))
        if do_devil<0:
            devil.append(TextClip(f"{angel_do[0]}.{angel_do[1][0:2]} ({sub_value[0]}.{sub_value[1][0:2]}%)↓ today",font="Arial Black",fontsize = 720//35,color="gray").set_pos((30,90)))
            
        devil.append(TextClip(f"{name_out}",font="Arial Black",fontsize = 720//35,color="gray").set_pos((470,40)))
        devil.append(ImageClip(cv2.resize(cv2.cvtColor(cv2.imread(r"static/devil_graph_out/devil_.jpg"),cv2.COLOR_BGR2RGB),(790,400))).set_position(('center',120)))
        devil.append(ImageClip(cv2.resize(cv2.cvtColor(cv2.imread(r"static/devil_graph_out/devil.jpg"),cv2.COLOR_BGR2RGB),(790,400))).set_position(('center',510)))
        for i,j in zip(devil_list_out,range(len(devil_list_out))):
            count_+=1
            z=[s for s in str(i[1]).split('.')]
            print(z)
            if len(z)>1: 
                if len(z[1])>4:
                    z[1]=z[1][0:4]
            devil.append(TextClip(f"{i[0].capitalize()}",font="Arial Black",fontsize = 720//35,color="gray").set_pos((x,count)))
            devil.append(TextClip(f"{'.'.join(z)}",font="Arial Black",fontsize = 720//35,color="gray").set_pos((x+100,count)))
            x+=220
            if count_%2==0 and count_!=0:
                count+=40
                x=30
        #↓
        devil.append(TextClip(f"our predicted value".capitalize(),font="Arial Black",fontsize = 720//35,color="gray").set_pos((470,930)))
        prid_devil=str(pridict).split('.')
        if pridict>devil_list_out[3][1]:
            devil.append(TextClip(f"{prid_devil[0]}.{prid_devil[1][0:2]}↑",font="Arial Black",fontsize = 720//15,color="gray").set_pos((480,960)))
        if pridict<devil_list_out[3][1]:
            devil.append(TextClip(f"{prid_devil[0]}.{prid_devil[1][0:2]}↓",font="Arial Black",fontsize = 720//15,color="gray").set_pos((480,960)))
            
        CompositeVideoClip([clip]+devil).save_frame(f"static/frame_devil/{devil_out}.png",t=1)

#list_outer=["WIPRO.NS","TCS.NS","LICI.NS","HDFCBANK.NS","RELIANCE.NS","IRCTC.NS","INFY.NS","SBIN.NS","HINDUNILVR.NS","ITC.NS"]
list_outer=["BHARTIARTL.NS","SUNPHARMA.NS","LT.NS","MARUTI.NS","ONGC.NS","AXISBANK.NS","TITAN.NS"]###
#list_outer=["ADANIENT.NS","ADANIGREEN.NS","ADANIPORTS.NS","ADANIPOWER.NS","ADANIENSOL.NS","JSWSTEEL.NS","TATAMOTORS.NS","TATASTEEL.NS","TATAPOWER.NS","TATACONSUM.NS"]
#list_outer=["JIOFIN.NS","ATGL.NS","EICHERMOT.NS","HEROMOTOCO.NS","BOSCHLTD.NS","TECHM.NS","GAIL.NS","BRITANNIA.NS"]
#list_outer=["ICICIBANK.NS","PNB.NS","INDUSINDBK.NS","UNIONBANK.NS","CANBK.NS","UCOBANK.NS","COALINDIA.NS","IRFC.NS","IOC.NS","GAIL.NS"]        
for i in range(len(list_outer)):
    devil_angel_out=devil_class.devil_connect_out(list_outer[i])
    devil_class.devil_front_out(devil_angel_out[0],devil_angel_out[1],devil_angel_out[2][0],devil_angel_out[3],i,devil_angel_out[4],)
concatenate_videoclips([ImageClip(f"static/frame_devil/{i}.png").set_duration(5) for i in range(len(list_outer))]).write_videofile("devil_angel.mp4",fps=20)
