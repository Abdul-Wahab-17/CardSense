package com.app.card_sense.controllers;

import java.io.OutputStream;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;
import org.opencv.videoio.Videoio;  
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import jakarta.servlet.http.HttpServletResponse;
import nu.pattern.OpenCV;

@RestController
public class CameraController {

    static {
        OpenCV.loadLocally(); // load once
    }

    @GetMapping(value = "/video", produces = "multipart/x-mixed-replace; boundary=frame")
    public void streamVideo(HttpServletResponse response) throws Exception {
 VideoCapture camera = new VideoCapture(0);

        // Force MJPEG (your camera supports MJPG)
        camera.set(Videoio.CAP_PROP_FOURCC, VideoWriter.fourcc('M','J','P','G'));
        camera.set(Videoio.CAP_PROP_FRAME_WIDTH, 1280);
        camera.set(Videoio.CAP_PROP_FRAME_HEIGHT, 720);
if (!camera.isOpened()) {
    System.out.println("Failed to open camera");
}

        Mat frame = new Mat();
        MatOfByte buffer = new MatOfByte();

        OutputStream out = response.getOutputStream();

        while (true) {
            camera.read(frame);

            if (!frame.empty()) {

                Imgcodecs.imencode(".jpg", frame, buffer);

                out.write(("--frame\r\n").getBytes());
                out.write(("Content-Type: image/jpeg\r\n\r\n").getBytes());
                out.write(buffer.toArray());
                out.write(("\r\n").getBytes());

                out.flush();
            }

            Thread.sleep(30); // ~30 FPS
        }
    }
}