package com.neonline.game;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Bundle;
import android.util.Log;

import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {

    private static final String TAG = "MainActivity";

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        new Thread(() -> {
            try {
                MlsdDetector detector = new MlsdDetector(MainActivity.this);

                Bitmap test = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888);
                Canvas c = new Canvas(test);
                c.drawColor(Color.WHITE);
                Paint p = new Paint();
                p.setColor(Color.BLACK);
                p.setStrokeWidth(6);
                c.drawLine(50, 50, 450, 300, p);
                c.drawLine(100, 400, 400, 420, p);
                c.drawLine(200, 100, 220, 500, p);

                Log.i(TAG, "test bitmap ready, running inference...");
                long t0 = System.currentTimeMillis();
                detector.testInference(test);
                Log.i(TAG, "inference total time: " + (System.currentTimeMillis() - t0) + " ms");

            } catch (Exception e) {
                Log.e(TAG, "error", e);
            }
        }).start();
    }
}
