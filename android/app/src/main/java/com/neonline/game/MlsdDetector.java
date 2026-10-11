package com.neonline.game;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * M-LSD tflite 模型封装。
 * 暂时只做：加载 + 用一张 512x512 的图跑一次推理，把输出 shape 打出来。
 */
public class MlsdDetector {

    private static final String TAG = "MlsdDetector";
    private static final String MODEL_FILE = "mlsd_fp16.tflite";
    private static final int INPUT_SIZE = 512;

    private Interpreter interpreter;

    public MlsdDetector(Context context) throws IOException {
        MappedByteBuffer model = loadModelFile(context, MODEL_FILE);
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        interpreter = new Interpreter(model, options);
        Log.i(TAG, "TFLite model loaded");

        // 打印输入输出 shape
        int[] inputShape = interpreter.getInputTensor(0).shape();
        Log.i(TAG, "input shape: " + java.util.Arrays.toString(inputShape));
        for (int i = 0; i < interpreter.getOutputTensorCount(); i++) {
            int[] shape = interpreter.getOutputTensor(i).shape();
            Log.i(TAG, "output[" + i + "] shape: " + java.util.Arrays.toString(shape));
        }
    }

    /**
     * 用一张 Bitmap 跑一次推理，只打印输出，不解析。
     */
    public void testInference(Bitmap bitmap) {
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);

        // 准备输入 [1, 512, 512, 4] float32, 0~1
        ByteBuffer input = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 4 * 4);
        input.order(ByteOrder.nativeOrder());

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        for (int p : pixels) {
            float r = ((p >> 16) & 0xFF) / 255f;
            float g = ((p >> 8) & 0xFF) / 255f;
            float b = (p & 0xFF) / 255f;
            float a = ((p >> 24) & 0xFF) / 255f;
            input.putFloat(r);
            input.putFloat(g);
            input.putFloat(b);
            input.putFloat(a);
        }
        input.rewind();

        // 输出占位（用 Map 做通用输出）
        java.util.Map<Integer, Object> outputs = new java.util.HashMap<>();
        // 按模型实际输出 shape 预分配
        // 暂时用 float 和 int 两种都试
        // 先跑一次看看
        try {
            interpreter.runForMultipleInputsOutputs(
                new Object[]{ input },
                outputs
            );
        } catch (Exception e) {
            Log.e(TAG, "inference failed", e);
            return;
        }

        Log.i(TAG, "inference done, outputs=" + outputs.size());
        for (java.util.Map.Entry<Integer, Object> e : outputs.entrySet()) {
            Object v = e.getValue();
            if (v instanceof float[][][]) {
                float[][][] arr = (float[][][]) v;
                Log.i(TAG, "output[" + e.getKey() + "] float[][][]: " +
                    arr.length + "x" + arr[0].length + "x" + arr[0][0].length);
            } else if (v instanceof float[][]) {
                float[][] arr = (float[][]) v;
                Log.i(TAG, "output[" + e.getKey() + "] float[][]: " +
                    arr.length + "x" + arr[0].length);
            } else if (v instanceof int[][][]) {
                int[][][] arr = (int[][][]) v;
                Log.i(TAG, "output[" + e.getKey() + "] int[][][]: " +
                    arr.length + "x" + arr[0].length + "x" + arr[0][0].length);
            } else {
                Log.i(TAG, "output[" + e.getKey() + "] type=" + v.getClass().getSimpleName());
            }
        }
    }

    private MappedByteBuffer loadModelFile(Context context, String name) throws IOException {
        AssetFileDescriptor fd = context.getAssets().openFd(name);
        FileInputStream is = new FileInputStream(fd.getFileDescriptor());
        FileChannel channel = is.getChannel();
        return channel.map(FileChannel.MapMode.READ_ONLY,
            fd.getStartOffset(), fd.getDeclaredLength());
    }
}
