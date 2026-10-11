package com.neonline.game;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MlsdDetector {

    private static final String TAG = "MlsdDetector";
    private static final String MODEL_FILE = "mlsd_fp16.tflite";
    private static final int INPUT_SIZE = 512;
    private static final int OUTPUT_SIZE = 256;
    private static final int OUTPUT_CHANNELS = 9;

    private static final float CENTER_THRESH = 1.0f;
    private static final float DIS_SCALE = OUTPUT_SIZE / 2f;

    private Interpreter interpreter;
    private float[][][][] outputBuffer;

    public static class Line {
        public float x1, y1, x2, y2;
    }

    public MlsdDetector(Context context) throws IOException {
        MappedByteBuffer model = loadModelFile(context, MODEL_FILE);
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        interpreter = new Interpreter(model, options);

        Log.i(TAG, "input shape: " + Arrays.toString(interpreter.getInputTensor(0).shape()));
        Log.i(TAG, "output shape: " + Arrays.toString(interpreter.getOutputTensor(0).shape()));

        outputBuffer = new float[1][OUTPUT_CHANNELS][OUTPUT_SIZE][OUTPUT_SIZE];
    }

    public void testInference(Bitmap bitmap) {
        List<Line> lines = detect(bitmap);
        Log.i(TAG, "detected " + lines.size() + " lines");
        for (int i = 0; i < Math.min(10, lines.size()); i++) {
            Line l = lines.get(i);
            Log.i(TAG, "line " + i + ": (" + l.x1 + "," + l.y1 + ") -> (" + l.x2 + "," + l.y2 + ")");
        }
    }

    public List<Line> detect(Bitmap bitmap) {
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);

        ByteBuffer input = ByteBuffer.allocateDirect(1 * 4 * INPUT_SIZE * INPUT_SIZE * 4);
        input.order(ByteOrder.nativeOrder());

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        for (int p : pixels) input.putFloat(((p >> 16) & 0xFF) / 255f);
        for (int p : pixels) input.putFloat(((p >> 8) & 0xFF) / 255f);
        for (int p : pixels) input.putFloat((p & 0xFF) / 255f);
        for (int p : pixels) input.putFloat(((p >> 24) & 0xFF) / 255f);
        input.rewind();

        interpreter.run(input, outputBuffer);

        // ============ 诊断：每个通道的统计 ============
        Log.i(TAG, "=== channel diagnostics ===");
        for (int c = 0; c < OUTPUT_CHANNELS; c++) {
            int cnt = 0;
            float mx = -9999, mn = 9999;
            for (int y = 0; y < OUTPUT_SIZE; y++) {
                for (int x = 0; x < OUTPUT_SIZE; x++) {
                    float v = outputBuffer[0][c][y][x];
                    if (v > mx) mx = v;
                    if (v < mn) mn = v;
                    if (v > 1.0f) cnt++;
                }
            }
            Log.i(TAG, "ch " + c + ": min=" + mn + " max=" + mx + " cnt>1=" + cnt);
        }
        // ==============================================

        List<Line> lines = new ArrayList<>();
        float[][][] out = outputBuffer[0];

        for (int y = 0; y < OUTPUT_SIZE; y++) {
            for (int x = 0; x < OUTPUT_SIZE; x++) {
                float centerLogit = out[0][y][x];
                if (centerLogit < CENTER_THRESH) continue;

                float[] p1 = decodeDisplacement(out, x, y, 1);
                float[] p2 = decodeDisplacement(out, x, y, 5);

                float cx = x * 2f, cy = y * 2f;
                float x1 = cx + p1[0] * 2f;
                float y1 = cy + p1[1] * 2f;
                float x2 = cx + p2[0] * 2f;
                float y2 = cy + p2[1] * 2f;

                Line l = new Line();
                l.x1 = x1; l.y1 = y1; l.x2 = x2; l.y2 = y2;
                lines.add(l);
            }
        }

        return lines;
    }

    private float[] decodeDisplacement(float[][][] out, int x, int y, int startCh) {
        float[] vals = new float[4];
        float maxV = -9999;
        for (int i = 0; i < 4; i++) {
            vals[i] = out[startCh + i][y][x];
            if (vals[i] > maxV) maxV = vals[i];
        }
        float sumExp = 0;
        for (int i = 0; i < 4; i++) {
            vals[i] = (float) Math.exp(vals[i] - maxV);
            sumExp += vals[i];
        }
        for (int i = 0; i < 4; i++) vals[i] /= sumExp;

        float dx = (vals[0] - vals[1]) * DIS_SCALE;
        float dy = (vals[2] - vals[3]) * DIS_SCALE;
        return new float[]{ dx, dy };
    }

    private MappedByteBuffer loadModelFile(Context context, String name) throws IOException {
        AssetFileDescriptor fd = context.getAssets().openFd(name);
        FileInputStream is = new FileInputStream(fd.getFileDescriptor());
        FileChannel channel = is.getChannel();
        return channel.map(FileChannel.MapMode.READ_ONLY,
            fd.getStartOffset(), fd.getDeclaredLength());
    }
}
