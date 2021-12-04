package com.example.facedet;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static String TAG = "CamActivity";
    private final int PERMISSIONS_READ_CAMERA=1;


    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;
    private Mat mRgba;
    private Mat mGray;

    private CascadeClassifier cascadeClassifier;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        cameraBridgeViewBase = (JavaCameraView)findViewById(R.id.CameraView);
        cameraBridgeViewBase.setVisibility(View.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);

        // Loading the model into the application
        try{
            InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE); // Create a folder
            File mCascadeFile = new File(cascadeDir,"haarcascade_frontalface_alt.xml"); //Creating a new file in that folder
            FileOutputStream os = new FileOutputStream(mCascadeFile);
            byte[] buffer = new byte[4096];
            int byteRead;
            //writing file from raw folder
            while((byteRead = is.read(buffer)) != -1){
                os.write(buffer, 0, byteRead);
            }
            is.close();
            os.close();

            // Load file from the folder created
            cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
        }
        catch (IOException e){
            Log.i(TAG, "Cascade file is not found");
        }


        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {

        } else {
            Log.d(TAG, "Permissions granted");
            cameraBridgeViewBase.setCameraPermissionGranted();
        }
        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                super.onManagerConnected(status);
                switch (status){
                    case BaseLoaderCallback.SUCCESS:
                        cameraBridgeViewBase.enableView();
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;
                }
            }
        };
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba=new Mat(height,width, CvType.CV_8UC4);
        mGray =new Mat(height,width, CvType.CV_8UC1);
    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba=inputFrame.rgba();
        //mGray=inputFrame.gray();

        // We pass mRgba to cascaderec

        mRgba = CascadeRec(mRgba);
        return mRgba;

    }

    private Mat CascadeRec(Mat mRgba) {
        // Rotate frame by -90 deg
        Core.flip(mRgba.t(), mRgba, 1);
        // Convert image to RGB
        Mat mRbg = new Mat();
        Imgproc.cvtColor(mRgba, mRbg,Imgproc.COLOR_RGBA2RGB);

        int height = mRbg.height();
        //Minimum size of face in frame
        int absoluteFaceSize = (int) (height*0.1);

        MatOfRect faces = new MatOfRect();
        if(cascadeClassifier !=null){
            cascadeClassifier.detectMultiScale(mRbg,faces,1.1,2,2, new Size(absoluteFaceSize,absoluteFaceSize), new Size());
        }

        Rect[] facesArray = faces.toArray();
        for(int i=0;i<facesArray.length;i++){
            //Draw bounding box
            Imgproc.rectangle(mRgba, facesArray[i].tl(),facesArray[i].br(),new Scalar(0,255,0), 2);
        }

        Core.flip(mRgba.t(), mRgba, 0);
        return mRgba;
    }

    @Override
    protected void onPostResume() {
        super.onPostResume();
        if (!OpenCVLoader.initDebug()){
            Toast.makeText(this, "Error", Toast.LENGTH_SHORT).show();
        }
        else
        {
            baseLoaderCallback.onManagerConnected(baseLoaderCallback.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (cameraBridgeViewBase != null){
            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraBridgeViewBase != null){
            cameraBridgeViewBase.disableView();
        }
    }
    @Override
    public void onRequestPermissionsResult(
            int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {

        // Ensure that this result is for the camera permission request
        if (requestCode == PERMISSIONS_READ_CAMERA) {
            // Check if the request was granted or denied
            if (grantResults.length > 0
                    && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // The request was granted -> tell the camera view
                cameraBridgeViewBase.setCameraPermissionGranted();
            } else {
                // The request was denied -> tell the user and exit the application
                Toast.makeText(this, "Camera permission required.",
                        Toast.LENGTH_LONG).show();
                this.finish();
            }
        } else {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }

}