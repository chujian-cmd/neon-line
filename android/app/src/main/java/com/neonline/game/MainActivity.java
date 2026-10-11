package com.neonline.game;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;

import com.getcapacitor.BridgeActivity;

import java.io.InputStream;

public class MainActivity extends BridgeActivity {

    private static final String TAG = "MainActivity";

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        new Thread(() -> {
            try {
                MlsdDetector detector = new MlsdDetector(MainActivity.this);

                InputStream is = getAssets().open("test.jpg");
                Bitmap test = BitmapFactory.decodeStream(is);
                is.close();

                if (test == null) {
                    Log.e(TAG, "failed to load test.jpg");
                    return;
                }
                Log.i(TAG, "test.jpg loaded: " + test.getWidth() + "x" + test.getHeight());

                long t0 = System.currentTimeMillis();
                detector.testInference(test);
                Log.i(TAG, "inference total time: " + (System.currentTimeMillis() - t0) + " ms");

            } catch (Exception e) {
                Log.e(TAG, "error", e);
            }
        }).start();
    }
}
