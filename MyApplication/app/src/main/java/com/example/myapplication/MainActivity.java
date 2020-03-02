package com.example.myapplication;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.app.ProgressDialog;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Display;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.tflite.ImageClassifier;

import org.opencv.android.CameraActivity;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.MSER;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import static java.lang.Thread.*;
import static org.opencv.imgproc.Imgproc.COLOR_RGB2GRAY;
import static org.opencv.imgproc.Imgproc.rectangle;

public class MainActivity extends Activity {

final static int CAPTURE_IMAGE_ACTIVITY_REQUEST_CODE = 1;
    private Scalar CONTOUR_COLOR;

Uri imageUri                      = null;
private TesseractOCR mTessOCR;
private ProgressDialog mProgressDialog;
    Bitmap mBitmap;
//static TextView imageDetails      = null;
public  static ImageView showImg  = null;
    MainActivity CameraActivity = null;
    Mat mat ;
    Mat outmat;
    Mat minimg;
    MatOfPoint2f approxCurve;
    TextView ocrtext;
    int threshold;
    Button doscan;
    MainActivity mainActivity;
    Bitmap bmp32;
    List<Mat> subimage ;
    int countsubimg = 0;
    Mat dst = null;
    Mat outgrey;
    private static final String TAG = "Opencv:tflite";


    private final Object lock = new Object();
    private boolean runClassifier = false;
    private Handler backgroundHandler;
    private HandlerThread backgroundThread;
    public ImageClassifier classifier;
    private static final String HANDLE_THREAD_NAME = "CameraBackground";
    private String ltext= "n";
    private TfliteTask tfliteTask;
    private boolean click = false;

    @Override
protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera_photo_capture);
    threshold = 100;
    String language = "eng";

    mTessOCR = new TesseractOCR(this, language);
    mProgressDialog = new ProgressDialog(MainActivity.this);
    OpenCVLoader.initDebug();
        CameraActivity = this;
    subimage = new ArrayList<>();

     //   imageDetails = (TextView) findViewById(R.id.imageDetails);
    doscan =(Button) findViewById(R.id.scan);
        showImg = (ImageView) findViewById(R.id.showImg);
    ocrtext = (TextView)findViewById(R.id.ocrtext);
    mainActivity =this;
   tfliteTask = new TfliteTask();
   tfliteTask.execute();
final Button photo = (Button) findViewById(R.id.photo);
    doscan.setOnClickListener(new View.OnClickListener() {
        @Override
        public void onClick(View view) {

         //   if(countsubimg < subimage.size()) {
             //   Mat tempmat = subimage.get(countsubimg);
                Bitmap img_bitmap1 = Bitmap.createBitmap(outmat.cols(), outmat.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(outmat, img_bitmap1);
                mBitmap = img_bitmap1;

          //  Point pt = new Point(0,r.y + ((r.height + text.height) / 2));
           // Imgproc.putText(im, ltext, pt, 2, scale, new Scalar(255, 0, 0), thickness);
                //  countsubimg++;
         //   }

            showImg.setImageBitmap(mBitmap);
            click =true;

            doOCR(mBitmap);
        }
    });


        photo.setOnClickListener(new View.OnClickListener() {
public void onClick(View v) {

        /*************************** Camera Intent Start ************************/

        // Define the file-name to save photo taken by Camera activity

        String fileName = "Camera_Example.jpg";



        // Create parameters for Intent with filename

        ContentValues values = new ContentValues();

        values.put(MediaStore.Images.Media.TITLE, fileName);

        values.put(MediaStore.Images.Media.DESCRIPTION,"Image capture by camera");

        // imageUri is the current activity attribute, define and save it for later usage

        imageUri = getContentResolver().insert(
        MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);

        /**** EXTERNAL_CONTENT_URI : style URI for the "primary" external storage volume. ****/


        // Standard Intent action that can be sent to have the camera
        // application capture an image and return it.

        Intent intent = new Intent( MediaStore.ACTION_IMAGE_CAPTURE );

        intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);

        intent.putExtra(MediaStore.EXTRA_VIDEO_QUALITY, 1);

        startActivityForResult( intent, CAPTURE_IMAGE_ACTIVITY_REQUEST_CODE);

        /*************************** Camera Intent End ************************/


        }

        });
        }

    class TfliteTask extends AsyncTask<Void, Void, Void> {

        @Override
        protected void onPreExecute() {
            super.onPreExecute();

        }

        @Override
        protected Void doInBackground(Void... params) {

            try {
                classifier = new ImageClassifier(mainActivity);
                Log.e(TAG, "*************************");
            } catch (IOException e) {
                Log.e(TAG, "Failed to initialize an image classifier.");
            }




            return null;
        }

        @Override
        protected void onPostExecute(Void result) {
            super.onPostExecute(result);
            startBackgroundThread();
            //myMap.addMarker(new MarkerOptions().position(new LatLng(dLat, dLon)).title(name));
            //  dialog.dismiss();

        }
    }
    /** Classifies a frame from the preview stream. */
    private void classifyFrame() {
        if (classifier == null || mainActivity == null ) {
            showToast("Uninitialized Classifier or invalid context.");
            return;
        }
        //   Bitmap bitmap =
        //          textureView.getBitmap(ImageClassifier.DIM_IMG_SIZE_X, ImageClassifier.DIM_IMG_SIZE_Y);click
        if(click) {
            if (mBitmap != null ) {
                String textToShow = classifier.classifyFrame(mBitmap);
                mBitmap.recycle();
                showToast(textToShow + "*** " + mBitmap.getWidth());
                click = false;
            }
        }
    }



    private void showToast(final String text) {
        final Activity activity = mainActivity;
        if (activity != null) {
            activity.runOnUiThread(

                    new Runnable() {
                        @Override
                        public void run() {
                            ltext = text;
                            ocrtext.setText(text);
                            //  textView.setText(text);
                        }
                    });
        }
    }

    /** Starts a background thread and its {@link Handler}. */
    private void startBackgroundThread() {
        backgroundThread = new HandlerThread(HANDLE_THREAD_NAME);
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
        synchronized (lock) {
            runClassifier = true;
        }
        backgroundHandler.post(periodicClassify);
    }

    /** Stops the background thread and its {@link Handler}. */
    private void stopBackgroundThread() {
        backgroundThread.quitSafely();
        try {
            backgroundThread.join();
            backgroundThread = null;
            backgroundHandler = null;
            synchronized (lock) {
                runClassifier = false;
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private Runnable periodicClassify =
            new Runnable() {
                @Override
                public void run() {
                    synchronized (lock) {
                        if (runClassifier) {
                            classifyFrame();
                        }
                    }
                    backgroundHandler.post(periodicClassify);
                }
            };


    @Override
protected void onActivityResult( int requestCode, int resultCode, Intent data)
        {
        if ( requestCode == CAPTURE_IMAGE_ACTIVITY_REQUEST_CODE) {

        if ( resultCode == RESULT_OK) {

        /*********** Load Captured Image And Data Start ****************/

        String imageId = convertImageUriToFile( imageUri,CameraActivity);


        //  Create and excecute AsyncTask to load capture image

        new LoadImagesFromSDCard().execute(""+imageId);

        /*********** Load Captured Image And Data End ****************/


        } else if ( resultCode == RESULT_CANCELED) {

        Toast.makeText(this, " Picture was not taken ", Toast.LENGTH_SHORT).show();
        } else {

        Toast.makeText(this, " Picture was not taken ", Toast.LENGTH_SHORT).show();
        }
        }
        }







    private void doOCR(final Bitmap ibitmap) {
        if (mProgressDialog == null) {
            mProgressDialog = ProgressDialog.show(mainActivity, "Processing",
                    "Doing OCR...", true);
            mProgressDialog.show();
        } else {
            mProgressDialog.show();
        }
        new Thread(new Runnable() {
            public void run() {
                final String srcText = mTessOCR.getOCRResult(ibitmap);
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {

                        if (srcText != null && !srcText.equals("")) {
                            ocrtext.setText(srcText);
                        }
                        mProgressDialog.dismiss();
                    }
                });
            }
        }).start();
    }












    /************ Convert Image Uri path to physical path **************/

public static String convertImageUriToFile ( Uri imageUri, Activity activity )  {

        Cursor cursor = null;
        int imageID = 0;

        try {

        /*********** Which columns values want to get *******/
        String [] proj={
        MediaStore.Images.Media.DATA,
        MediaStore.Images.Media._ID,
        MediaStore.Images.Thumbnails._ID,
        MediaStore.Images.ImageColumns.ORIENTATION
        };

        cursor = activity.managedQuery(

        imageUri,         //  Get data for specific image URI
        proj,             //  Which columns to return
        null,             //  WHERE clause; which rows to return (all rows)
        null,             //  WHERE clause selection arguments (none)
        null              //  Order-by clause (ascending by name)

        );

        //  Get Query Data

        int columnIndex = cursor.getColumnIndexOrThrow(MediaStore.Images.Media._ID);
        int columnIndexThumb = cursor.getColumnIndexOrThrow(MediaStore.Images.Thumbnails._ID);
        int file_ColumnIndex = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);

        //int orientation_ColumnIndex = cursor.
        //    getColumnIndexOrThrow(MediaStore.Images.ImageColumns.ORIENTATION);

        int size = cursor.getCount();

        /*******  If size is 0, there are no images on the SD Card. *****/

        if (size == 0) {


  //      imageDetails.setText("No Image");
        }
        else
        {

        int thumbID = 0;
        if (cursor.moveToFirst()) {

        /**************** Captured image details ************/

        /*****  Used to show image on view in LoadImagesFromSDCard class ******/
        imageID     = cursor.getInt(columnIndex);

        thumbID     = cursor.getInt(columnIndexThumb);

        String Path = cursor.getString(file_ColumnIndex);

        //String orientation =  cursor.getString(orientation_ColumnIndex);

        String CapturedImageDetails = " CapturedImageDetails : \n\n"
        +" ImageID :"+imageID+"\n"
        +" ThumbID :"+thumbID+"\n"
        +" Path :"+Path+"\n";
            Log.i("TAG_____________", CapturedImageDetails);

        // Show Captured Image detail on activity
//        imageDetails.setText( CapturedImageDetails );

        }
        }
        } finally {
        if (cursor != null) {
        cursor.close();
        }
        }

        // Return Captured Image ImageID ( By this ImageID Image will load from sdcard )

        return ""+imageID;
        }


/**
 * Async task for loading the images from the SD card.
 *
 * @author Android Example
 *
 */

// Class with extends AsyncTask class

public class LoadImagesFromSDCard  extends AsyncTask<String, Void, Void> {





    protected void onPreExecute() {
        /****** NOTE: You can call UI Element here. *****/

        // Progress Dialog
   //     Dialog.setMessage(" Loading image from Sdcard..");
   //     Dialog.show();
    }


    // Call after onPreExecute method
    protected Void doInBackground(String... urls) {
        Mat bwIMG, hsvIMG, lrrIMG, urrIMG, dsIMG, usIMG, cIMG, hovIMG;
        Bitmap bitmap = null;
        Bitmap newBitmap = null;
        Uri uri = null;
        Mat img = null;
        Mat gray = null;


        try {

            /**  Uri.withAppendedPath Method Description
             * Parameters
             *    baseUri  Uri to append path segment to
             *    pathSegment  encoded path segment to append
             * Returns
             *    a new Uri based on baseUri with the given segment appended to the path
             */
            Display display = getWindowManager().getDefaultDisplay();
            int width = display.getWidth();
            int height = display.getHeight();



            uri = Uri.withAppendedPath(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "" + urls[0]);

            /**************  Decode an input stream into a bitmap. *********/
            bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(uri));

            if (bitmap != null) {

                /********* Creates a new bitmap, scaled from an existing bitmap. ***********/

                newBitmap = Bitmap.createScaledBitmap(bitmap, width, height, true);
                bmp32 = newBitmap.copy(Bitmap.Config.ARGB_8888, true);
               bitmap.recycle();

                if (newBitmap != null) {
                    img = new Mat (newBitmap.getWidth(), newBitmap.getHeight(), CvType.CV_8UC1);
                    Utils.bitmapToMat(newBitmap, img);


                    bwIMG = new Mat();
                    dsIMG = new Mat();
                    hsvIMG = new Mat();
                    lrrIMG = new Mat();
                    urrIMG = new Mat();
                    usIMG = new Mat();
                    cIMG = new Mat();
                    hovIMG = new Mat();
                    gray = new Mat();
                    dst = new Mat();
                    minimg = new Mat();
                    outgrey = new Mat();
                    approxCurve = new MatOfPoint2f();


                    Imgproc.cvtColor(img, gray, COLOR_RGB2GRAY);
                    gray.copyTo(outgrey);
                    Imgproc.cvtColor(img, dst, Imgproc.COLOR_RGB2RGBA);



                    Imgproc.pyrDown(gray, dsIMG, new Size(gray.cols() / 2, gray.rows() / 2));
                    Imgproc.pyrUp(dsIMG, usIMG, gray.size());

                    Imgproc.Canny(usIMG, bwIMG, 0, threshold);

                    Imgproc.dilate(bwIMG, bwIMG, new Mat(), new Point(-1, 1), 1);

                    List<MatOfPoint> contours = new ArrayList<MatOfPoint>();

                    cIMG = bwIMG.clone();

                    Imgproc.findContours(cIMG, contours, hovIMG, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);


                    for (MatOfPoint cnt : contours) {

                        MatOfPoint2f curve = new MatOfPoint2f(cnt.toArray());

                        Imgproc.approxPolyDP(curve, approxCurve, 0.02 * Imgproc.arcLength(curve, true), true);

                        int numberVertices = (int) approxCurve.total();

                        double contourArea = Imgproc.contourArea(cnt);

                        if (Math.abs(contourArea) < 100) {
                            continue;
                        }

                        //Rectangle detected
                        if (numberVertices >= 6 && numberVertices <= 8) {

                            List<Double> cos = new ArrayList<>();

                            for (int j = 2; j < numberVertices + 1; j++) {
                                cos.add(angle(approxCurve.toArray()[j % numberVertices], approxCurve.toArray()[j - 2], approxCurve.toArray()[j - 1]));

                            }

                            Collections.sort(cos);

                            double mincos = cos.get(0);
                            double maxcos = cos.get(cos.size() - 1);

                            if (numberVertices == 6 && mincos >= -0.6 && maxcos <= 0.1) {
                                //Imgproc.drawContours(dst, contours, -1, new Scalar(0, 0, 255));
                                setLabel(dst, "X", cnt,outgrey);
                            }

                        }


                    }

                    Mat img_result = dst.clone();
                  //  Imgproc.Canny(dst, img_result, 80, 90);
                    Bitmap img_bitmap1 = Bitmap.createBitmap(dst.cols(), dst.rows(),Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(img_result, img_bitmap1);
                    mBitmap = img_bitmap1;



                }
            }






                    } catch (Exception e) {
            // Error fetching image, try to recover

            /********* Cancel execution of this task. **********/
           // cancel(true);
        }

        return null;
    }

    private void detectHex( Mat img){
        Mat gray = null,bwIMG, hsvIMG, lrrIMG, urrIMG, dsIMG, usIMG, cIMG, hovIMG;

      try{
        bwIMG = new Mat();
        dsIMG = new Mat();
        hsvIMG = new Mat();
        lrrIMG = new Mat();
        urrIMG = new Mat();
        usIMG = new Mat();
        cIMG = new Mat();
        hovIMG = new Mat();
        gray = new Mat();
        dst = new Mat();
        minimg = new Mat();
        approxCurve = new MatOfPoint2f();


        Imgproc.cvtColor(img, gray, COLOR_RGB2GRAY);
        Imgproc.cvtColor(img, dst, Imgproc.COLOR_RGB2RGBA);



        Imgproc.pyrDown(gray, dsIMG, new Size(gray.cols() / 2, gray.rows() / 2));
        Imgproc.pyrUp(dsIMG, usIMG, gray.size());

        Imgproc.Canny(usIMG, bwIMG, 0, threshold);

        Imgproc.dilate(bwIMG, bwIMG, new Mat(), new Point(-1, 1), 1);

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();

        cIMG = bwIMG.clone();

        Imgproc.findContours(cIMG, contours, hovIMG, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);


        for (MatOfPoint cnt : contours) {

            MatOfPoint2f curve = new MatOfPoint2f(cnt.toArray());

            Imgproc.approxPolyDP(curve, approxCurve, 0.02 * Imgproc.arcLength(curve, true), true);

            int numberVertices = (int) approxCurve.total();

            double contourArea = Imgproc.contourArea(cnt);

            if (Math.abs(contourArea) < 100) {
                continue;
            }

            //Rectangle detected
            if (numberVertices >= 6 && numberVertices <= 8) {

                List<Double> cos = new ArrayList<>();

                for (int j = 2; j < numberVertices + 1; j++) {
                    cos.add(angle(approxCurve.toArray()[j % numberVertices], approxCurve.toArray()[j - 2], approxCurve.toArray()[j - 1]));

                }

                Collections.sort(cos);

                double mincos = cos.get(0);
                double maxcos = cos.get(cos.size() - 1);

                if (numberVertices == 6 && mincos >= -0.6 && maxcos <= 0.1) {
                    Imgproc.drawContours(dst, contours, -1, new Scalar(0, 0, 255));
                 //   setLabel(dst, "X", cnt);
                }

            }


        }

        Mat img_result = dst.clone();
        //  Imgproc.Canny(dst, img_result, 80, 90);
        Bitmap img_bitmap1 = Bitmap.createBitmap(dst.cols(), dst.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img_result, img_bitmap1);
        mBitmap = img_bitmap1;


} catch (Exception e) {
        // Error fetching image, try to recover

        /********* Cancel execution of this task. **********/
        // cancel(true);
        }
    }

    private  double angle(Point pt1, Point pt2, Point pt0) {
        double dx1 = pt1.x - pt0.x;
        double dy1 = pt1.y - pt0.y;
        double dx2 = pt2.x - pt0.x;
        double dy2 = pt2.y - pt0.y;
        return (dx1 * dx2 + dy1 * dy2) / Math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
    }

    private Mat mser(Mat inputim){
        Mat   mRgba = new Mat();
        Mat mGray = new Mat();

        Imgproc.cvtColor( inputim, mRgba,COLOR_RGB2GRAY);
        Imgproc.threshold(inputim, mGray ,120, 255,Imgproc.THRESH_BINARY);


//        MSER mser = MSER.create();
//        List<MatOfPoint> msers = new ArrayList<>();
//        MatOfRect bboxes = new MatOfRect();
//        mser.detectRegions(image, msers, bboxes);

        CONTOUR_COLOR = new Scalar(255);
        MatOfKeyPoint keypoint = new MatOfKeyPoint();
        List<KeyPoint> listpoint = new ArrayList<KeyPoint>();
        KeyPoint kpoint = new KeyPoint();
        Mat mask = Mat.zeros(mGray.size(), CvType.CV_8UC1);
        int rectanx1;
        int rectany1;
        int rectanx2;
        int rectany2;

        //
        Scalar zeos = new Scalar(0, 0, 0);
        List<MatOfPoint> contour1 = new ArrayList<MatOfPoint>();
        List<MatOfPoint> contour2 = new ArrayList<MatOfPoint>();
        Mat kernel = new Mat(1, 50, CvType.CV_8UC1, Scalar.all(255));
        Mat morbyte = new Mat();
        Mat hierarchy = new Mat();

        Rect rectan2 = new Rect();//
        Rect rectan3 = new Rect();//
        int imgsize = mRgba.height() * mRgba.width();

        //

        MSER detector = MSER.create();
        detector.detect(mGray, keypoint);
        listpoint = keypoint.toList();
        //






        for (int ind = 0; ind < listpoint.size(); ind++) {
            kpoint = listpoint.get(ind);
            rectanx1 = (int) (kpoint.pt.x - 0.5 * kpoint.size);
            rectany1 = (int) (kpoint.pt.y - 0.5 * kpoint.size);
            // rectanx2 = (int) (kpoint.pt.x + 0.5 * kpoint.size);
            // rectany2 = (int) (kpoint.pt.y + 0.5 * kpoint.size);
            rectanx2 = (int) (kpoint.size);
            rectany2 = (int) (kpoint.size);
            if (rectanx1 <= 0)
                rectanx1 = 1;
            if (rectany1 <= 0)
                rectany1 = 1;
            if ((rectanx1 + rectanx2) > mGray.width())
                rectanx2 = mGray.width() - rectanx1;
            if ((rectany1 + rectany2) > mGray.height())
                rectany2 = mGray.height() - rectany1;
            Rect rectant = new Rect(rectanx1, rectany1, rectanx2, rectany2);
            Mat roi = new Mat(mask, rectant);
            roi.setTo(CONTOUR_COLOR);

        }

        Imgproc.morphologyEx(mask, morbyte, Imgproc.MORPH_DILATE, kernel);
        Imgproc.findContours(morbyte, contour2, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
        for (int ind = 0; ind < contour2.size(); ind++) {
            rectan3 = Imgproc.boundingRect(contour2.get(ind));
            if (rectan3.area() > 0.5 * imgsize || rectan3.area() < 100
                    || rectan3.width / rectan3.height < 2) {
                Mat roi = new Mat(morbyte, rectan3);
                roi.setTo(zeos);

            } else

                rectangle(mRgba, rectan3.br(), rectan3.tl(),
                        CONTOUR_COLOR);


             return mRgba;
        }
        return mRgba;
    }

    private void setLabel(Mat im, String label, MatOfPoint contour, Mat imo) {
        // int fontface = Core.FONT_HERSHEY_SIMPLEX;
        double scale = 3;//0.4;
        int thickness = 3;//1;
        int[] baseline = new int[1];
        Size text = Imgproc.getTextSize(label, 3, scale, thickness, baseline);
        Rect r = Imgproc.boundingRect(contour);
        if(r.width >40 && r.height >40) {
            Mat miniMat = im.submat(r);
            ;
            minimg = miniMat;
            //  mser(minimg);
            //   Imgproc.cvtColor(miniMat, minimg, COLOR_RGB2GRAY);
            // Imgproc.threshold(minimg, minimg, 90, 255,Imgproc.THRESH_BINARY);
            // using an elliptical kernel
//        final Size kernelSize = new Size(6, 6);
            //   final Point anchor = new Point(-1, -1);
//        final int iterations = 3;

            //     Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, kernelSize);
            //     Imgproc.erode(minimg, minimg, kernel, anchor, iterations);
            //     Imgproc.dilate(minimg, minimg, kernel, anchor, iterations);

            // Imgproc.d
            //  Imgproc.blur(minimg,minimg,)
            outmat = minimg;
            // subimage.add(minimg);

            Imgproc.rectangle(
                    im,                    //Matrix obj of the image
                    new Point(r.x, r.y),        //p1
                    new Point(r.x + r.width, r.y + r.height),       //p2
                    new Scalar(0, 0, 255),     //Scalar object for color
                    5                        //Thickness of the line
            );
            //  detectHex(outmat);
            // Imgproc.drawContours(outmat, contour, -1, new Scalar(0, 0, 255));
            Point pt = new Point(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));
            Imgproc.putText(im, label, pt, 3, scale, new Scalar(255, 0, 0), thickness);
        }
    }

    protected void onPostExecute(Void unused) {

        // NOTE: You can call UI Element here.

        // Close progress dialog
   //     Dialog.dismiss();

       if(mBitmap != null)
        {

            // Set Image to ImageView

          showImg.setImageBitmap(mBitmap);
        }

    }

}

}
