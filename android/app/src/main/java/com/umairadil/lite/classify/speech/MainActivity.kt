package com.umairadil.lite.classify.speech

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.content.res.AssetManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.*
import android.util.Log
import android.view.View
import android.view.ViewTreeObserver.OnGlobalLayoutListener
import android.widget.CompoundButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.annotation.RequiresApi
import androidx.appcompat.widget.SwitchCompat
import androidx.core.content.ContextCompat
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetBehavior.BottomSheetCallback
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import java.util.concurrent.locks.ReentrantLock

class MainActivity : Activity(), View.OnClickListener, CompoundButton.OnCheckedChangeListener {

    var recordingBuffer = ShortArray(RECORDING_LENGTH)
    var recordingOffset = 0
    var shouldContinue = true
    private var recordingThread: Thread? = null
    var shouldContinueRecognition = true
    private var recognitionThread: Thread? = null
    private val recordingBufferLock = ReentrantLock()
    private val labels: MutableList<String> = ArrayList()
    private val displayedLabels: MutableList<String> = ArrayList()
    private var recognizeCommands: RecognizeCommands? = null
    private var bottomSheetLayout: LinearLayout? = null
    private var gestureLayout: LinearLayout? = null
    private var sheetBehavior: BottomSheetBehavior<*>? = null
    private var tfLite: Interpreter? = null
    private var bottomSheetArrowImageView: ImageView? = null
    private var leftTextView: TextView? = null
    private var rightTextView: TextView? = null
    private var sampleRateTextView: TextView? = null
    private var inferenceTimeTextView: TextView? = null
    private var plusImageView: ImageView? = null
    private var minusImageView: ImageView? = null
    private var apiSwitchCompat: SwitchCompat? = null
    private var threadsTextView: TextView? = null
    private var lastProcessingTimeMs: Long = 0
    private val handler = Handler()
    private var selectedTextView: TextView? = null
    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null

    /** Options for configuring the Interpreter.  */
    private val tfliteOptions = Interpreter.Options()

    @RequiresApi(Build.VERSION_CODES.M)
    override fun onCreate(savedInstanceState: Bundle?) {
        // Set up the UI.
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        displayedLabels.add("1")
        displayedLabels.add("2")

        labels.add("1")
        labels.add("2")

        // Set up an object to smooth recognition results to increase accuracy.
        recognizeCommands = RecognizeCommands(
                labels,
                AVERAGE_WINDOW_DURATION_MS,
                DETECTION_THRESHOLD,
                SUPPRESSION_MS,
                MINIMUM_COUNT,
                MINIMUM_TIME_BETWEEN_SAMPLES_MS)
        val actualModelFilename = MODEL_FILENAME.split("file:///android_asset/".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()[1]

        tfLite = try {
            tfliteOptions.setNumThreads(4)

            Interpreter(loadModelFile(assets, actualModelFilename), tfliteOptions)
        } catch (e: Exception) {
            throw RuntimeException(e)
        }

        //TODO: Verify
        //tfLite?.resizeInput(0, intArrayOf(RECORDING_LENGTH, 1))

        // Start the recording and recognition threads.
        requestMicrophonePermission()

        sampleRateTextView = findViewById(R.id.sample_rate)
        inferenceTimeTextView = findViewById(R.id.inference_info)
        bottomSheetLayout = findViewById(R.id.bottom_sheet_layout)
        gestureLayout = findViewById(R.id.gesture_layout)
        sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout)
        bottomSheetArrowImageView = findViewById(R.id.bottom_sheet_arrow)
        threadsTextView = findViewById(R.id.threads)
        plusImageView = findViewById(R.id.plus)
        minusImageView = findViewById(R.id.minus)
        apiSwitchCompat = findViewById(R.id.api_info_switch)
        leftTextView = findViewById(R.id.left)
        rightTextView = findViewById(R.id.right)
        apiSwitchCompat?.setOnCheckedChangeListener(this)
        val vto = gestureLayout?.viewTreeObserver
        vto?.addOnGlobalLayoutListener(
                object : OnGlobalLayoutListener {
                    override fun onGlobalLayout() {
                        gestureLayout?.viewTreeObserver?.removeOnGlobalLayoutListener(this)
                        val height = gestureLayout?.measuredHeight
                        sheetBehavior?.peekHeight = height!!
                    }
                })
        sheetBehavior?.isHideable = false
        sheetBehavior?.setBottomSheetCallback(
                object : BottomSheetCallback() {
                    override fun onStateChanged(bottomSheet: View, newState: Int) {
                        when (newState) {
                            BottomSheetBehavior.STATE_HIDDEN -> {
                            }
                            BottomSheetBehavior.STATE_EXPANDED -> {
                                bottomSheetArrowImageView?.setImageResource(R.drawable.ic_baseline_arrow_drop_down_24)
                            }
                            BottomSheetBehavior.STATE_COLLAPSED -> {
                                bottomSheetArrowImageView?.setImageResource(R.drawable.ic_baseline_arrow_drop_up_24)
                            }
                            BottomSheetBehavior.STATE_DRAGGING -> {
                            }
                            BottomSheetBehavior.STATE_SETTLING -> bottomSheetArrowImageView?.setImageResource(R.drawable.ic_baseline_arrow_drop_up_24)
                        }
                    }

                    override fun onSlide(bottomSheet: View, slideOffset: Float) {}
                })
        plusImageView?.setOnClickListener(this)
        minusImageView?.setOnClickListener(this)
        sampleRateTextView?.text = "$SAMPLE_RATE Hz"
    }

    @RequiresApi(Build.VERSION_CODES.M)
    private fun requestMicrophonePermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO)
        } else {
            startRecording()
            startRecognition()
        }
    }

    override fun onRequestPermissionsResult(
            requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        if (requestCode == REQUEST_RECORD_AUDIO && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startRecording()
            startRecognition()
        }
    }

    @Synchronized
    fun startRecording() {
        if (recordingThread != null) {
            return
        }
        shouldContinue = true
        recordingThread = Thread(
                Runnable { record() })
        recordingThread!!.start()
    }

    @Synchronized
    fun stopRecording() {
        if (recordingThread == null) {
            return
        }
        shouldContinue = false
        recordingThread = null
    }

    private fun record() {
        Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)

        // Estimate the buffer size we'll need for this device.
        var bufferSize = AudioRecord.getMinBufferSize(
                SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
            bufferSize = SAMPLE_RATE * 2
        }
        val audioBuffer = ShortArray(bufferSize / 2)
        val record = AudioRecord(
                MediaRecorder.AudioSource.DEFAULT,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize)
        if (record.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(LOG_TAG, "Audio Record can't initialize!")
            return
        }
        record.startRecording()
        Log.v(LOG_TAG, "Start recording")

        // Loop, gathering audio data and copying it to a round-robin buffer.
        while (shouldContinue) {
            val numberRead = record.read(audioBuffer, 0, audioBuffer.size)
            val maxLength = recordingBuffer.size
            val newRecordingOffset = recordingOffset + numberRead
            val secondCopyLength = Math.max(0, newRecordingOffset - maxLength)
            val firstCopyLength = numberRead - secondCopyLength
            // We store off all the data for the recognition thread to access. The ML
            // thread will copy out of this buffer into its own, while holding the
            // lock, so this should be thread safe.
            recordingBufferLock.lock()
            recordingOffset = try {
                System.arraycopy(audioBuffer, 0, recordingBuffer, recordingOffset, firstCopyLength)
                System.arraycopy(audioBuffer, firstCopyLength, recordingBuffer, 0, secondCopyLength)
                newRecordingOffset % maxLength
            } finally {
                recordingBufferLock.unlock()
            }
        }
        record.stop()
        record.release()
    }

    @Synchronized
    fun startRecognition() {
        if (recognitionThread != null) {
            return
        }
        shouldContinueRecognition = true
        recognitionThread = Thread(Runnable { recognize() })
        recognitionThread!!.start()
    }

    @Synchronized
    fun stopRecognition() {
        if (recognitionThread == null) {
            return
        }
        shouldContinueRecognition = false
        recognitionThread = null
    }

    private fun recognize() {
        Log.v(LOG_TAG, "Start recognition")
        val inputBuffer = ShortArray(RECORDING_LENGTH)
        val floatInputBuffer = Array(RECORDING_LENGTH) { FloatArray(1) }
        val outputScores = Array(1) { FloatArray(labels.size) }
        val sampleRateList = intArrayOf(SAMPLE_RATE)

        // Loop, grabbing recorded data and running the recognition model on it.
        while (shouldContinueRecognition) {
            val startTime = Date().time
            // The recording thread places data in this round-robin buffer, so lock to
            // make sure there's no writing happening and then copy it to our own
            // local version.
            recordingBufferLock.lock()
            try {
                val maxLength = recordingBuffer.size
                val firstCopyLength = maxLength - recordingOffset
                val secondCopyLength = recordingOffset
                System.arraycopy(recordingBuffer, recordingOffset, inputBuffer, 0, firstCopyLength)
                System.arraycopy(recordingBuffer, 0, inputBuffer, firstCopyLength, secondCopyLength)
            } finally {
                recordingBufferLock.unlock()
            }

            // We need to feed in float values between -1.0f and 1.0f, so divide the
            // signed 16-bit inputs.
            for (i in 0 until RECORDING_LENGTH) {
                floatInputBuffer[i][0] = inputBuffer[i] / 32767.0f
            }
            val inputArray = arrayOf<Any>(floatInputBuffer)
            val outputMap: MutableMap<Int, Any> = HashMap()
            outputMap[0] = outputScores

            // Run the model.
            tfLite?.runForMultipleInputsOutputs(inputArray, outputMap)

            // Use the smoother to figure out if we've had a real recognition event.
            val currentTime = System.currentTimeMillis()
            val result = recognizeCommands!!.processLatestResults(outputScores[0], currentTime)
            lastProcessingTimeMs = Date().time - startTime
            runOnUiThread {
                inferenceTimeTextView!!.text = "$lastProcessingTimeMs ms"

                // If we do have a new command, highlight the right list entry.
                if (!result.foundCommand.startsWith("_") && result.isNewCommand) {
                    var labelIndex = -1
                    for (i in labels.indices) {
                        if (labels[i] == result.foundCommand) {
                            labelIndex = i
                        }
                    }
                    when (labelIndex - 2) {
                        1 -> selectedTextView = leftTextView
                        2 -> selectedTextView = rightTextView
                    }
                    if (selectedTextView != null) {
                        selectedTextView!!.setBackgroundResource(R.drawable.round_corner_text_bg_selected)
                        val score = Math.round(result.score * 100).toString() + "%"
                        selectedTextView!!.text = """
                        ${selectedTextView!!.text}
                        $score
                        """.trimIndent()
                        selectedTextView!!.setTextColor(
                                resources.getColor(android.R.color.holo_orange_light))
                        handler.postDelayed(
                                {
                                    val origionalString = selectedTextView!!.text.toString().replace(score, "").trim { it <= ' ' }
                                    selectedTextView!!.text = origionalString
                                    selectedTextView!!.setBackgroundResource(
                                            R.drawable.round_corner_text_bg_unselected)
                                    selectedTextView!!.setTextColor(
                                            resources.getColor(android.R.color.darker_gray))
                                },
                                750)
                    }
                }
            }
            try {
                // We don't need to run too frequently, so snooze for a bit.
                Thread.sleep(MINIMUM_TIME_BETWEEN_SAMPLES_MS)
            } catch (e: InterruptedException) {
                // Ignore
            }
        }
        Log.v(LOG_TAG, "End recognition")
    }

    override fun onClick(v: View) {
        if (v.id == R.id.plus) {
            val threads = threadsTextView!!.text.toString().trim { it <= ' ' }
            var numThreads = threads.toInt()
            numThreads++
            threadsTextView!!.text = numThreads.toString()
            //            tfLite.setNumThreads(numThreads);
            val finalNumThreads = numThreads
            backgroundHandler!!.post { tfLite!!.setNumThreads(finalNumThreads) }
        } else if (v.id == R.id.minus) {
            val threads = threadsTextView!!.text.toString().trim { it <= ' ' }
            var numThreads = threads.toInt()
            if (numThreads == 1) {
                return
            }
            numThreads--
            threadsTextView!!.text = numThreads.toString()
            tfLite!!.setNumThreads(numThreads)
            val finalNumThreads = numThreads
            backgroundHandler!!.post { tfLite!!.setNumThreads(finalNumThreads) }
        }
    }

    override fun onCheckedChanged(buttonView: CompoundButton, isChecked: Boolean) {
        backgroundHandler!!.post { tfLite!!.setUseNNAPI(isChecked) }
        if (isChecked) apiSwitchCompat!!.text = "NNAPI" else apiSwitchCompat!!.text = "TFLITE"
    }

    private fun startBackgroundThread() {
        backgroundThread = HandlerThread(HANDLE_THREAD_NAME)
        backgroundThread!!.start()
        backgroundHandler = Handler(backgroundThread!!.looper)
    }

    private fun stopBackgroundThread() {
        backgroundThread!!.quitSafely()
        try {
            backgroundThread!!.join()
            backgroundThread = null
            backgroundHandler = null
        } catch (e: InterruptedException) {
            Log.e("amlan", "Interrupted when stopping background thread", e)
        }
    }

    override fun onResume() {
        super.onResume()
        startBackgroundThread()
    }

    override fun onStop() {
        super.onStop()
        stopBackgroundThread()
    }

    companion object {
        // Constants that control the behavior of the recognition code and model
        // settings. See the audio recognition tutorial for a detailed explanation of
        // all these, but you should customize them to match your training settings if
        // you are running your own model.
        private const val SAMPLE_RATE = 16000
        private const val SAMPLE_DURATION_MS = 1000
        private const val RECORDING_LENGTH = (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000)
        private const val AVERAGE_WINDOW_DURATION_MS: Long = 1000
        private const val DETECTION_THRESHOLD = 0.50f
        private const val SUPPRESSION_MS = 1500
        private const val MINIMUM_COUNT = 3
        private const val MINIMUM_TIME_BETWEEN_SAMPLES_MS: Long = 30
        private const val MODEL_FILENAME = "file:///android_asset/converted_model.tflite"

        // UI elements.
        private const val REQUEST_RECORD_AUDIO = 13
        private val LOG_TAG = MainActivity::class.java.simpleName

        /** Memory-map the model file in Assets.  */
        @Throws(IOException::class)
        private fun loadModelFile(assets: AssetManager, modelFilename: String): MappedByteBuffer {
            val fileDescriptor = assets.openFd(modelFilename)
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        }

        private const val HANDLE_THREAD_NAME = "CameraBackground"
    }
}