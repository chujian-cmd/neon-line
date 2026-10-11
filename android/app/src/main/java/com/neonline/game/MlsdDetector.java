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
import java.util.Arrays;

public class MlsdDetector {

    private static final String TAG = "MlsdDetector";
    private static final String MODEL_FILE = "mlsd_fp16.tflite";
    private static final int INPUT_SIZE = 512;
    private static final int OUTPUT_SIZE = 256;
    private static final int OUTPUT_CHANNELS = 9;

    private Interpreter interpreter;
    private float[][][][] outputBuffer;

    public MlsdDetector(Context context) throws IOException {
        MappedByteBuffer model = loadModelFile(context, MODEL_FILE);
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        interpreter = new Interpreter(model, options);

        int[] inputShape = interpreter.getInputTensor(0).shape();
        int[] outputShape = interpreter.getOutputTensor(0).shape();
        Log.i(TAG, "input shape: " + Arrays.toString(inputShape));
        Log.i(TAG, "output shape: " + Arrays.toString(outputShape));

        outputBuffer = new float[1][OUTPUT_CHANNELS][OUTPUT_SIZE][OUTPUT_SIZE];
    }

    public void testInference(Bitmap bitmap) {
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);

        // NCHW: [1, 4, 512, 512]
        ByteBuffer input = ByteBuffer.allocateDirect(1 * 4 * INPUT_SIZE * INPUT_SIZE * 4);
        input.order(ByteOrder.nativeOrder());

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        // 通道 R
        for (int p : pixels) input.putFloat(((p >> 16) & 0xFF) / 255f);
        // 通道 G
        for (int p : pixels) input.putFloat(((p >> 8) & 0xFF) / 255f);
        // 通道 B
        for (int p : pixels) input.putFloat((p & 0xFF) / 255f);
        // 通道 A
        for (int p : pixels) input.putFloat(((p >> 24) & 0xFF) / 255f);

        input.rewind();

        // 跑推理：输入 + 输出 buffer
        interpreter.run(input, outputBuffer);

        Log.i(TAG, "inference done");

        // 统计 TP map 中大于 0 的像素数，粗略判断有没有检测到东西
        int hitCount = 0;
        float maxVal = -9999;
        for (int y = 0; y < OUTPUT_SIZE; y++) {
            for (int x = 0; x < OUTPUT_SIZE; x++) {
                float v = outputBuffer[0][0][y][x];
                if (v > maxVal) maxVal = v;
                if (v > 0.5f) hitCount++;
            }
        }
        Log.i(TAG, "TP map max=" + maxVal + ", pixels>0.5: " + hitCount);
    }

    private MappedByteBuffer loadModelFile(Context context, String name) throws IOException {
        AssetFileDescriptor fd = context.getAssets().openFd(name);
        FileInputStream is = new FileInputStream(fd.getFileDescriptor());
        FileChannel channel = is.getChannel();
        return channel.map(FileChannel.MapMode.READ_ONLY,
            fd.getStartOffset(), fd.getDeclaredLength());
    }
}
