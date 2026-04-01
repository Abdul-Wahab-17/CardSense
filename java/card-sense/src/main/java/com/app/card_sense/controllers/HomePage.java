package com.app.card_sense.controllers;

import nu.pattern.OpenCV;
import org.opencv.core.Core;
import org.opencv.videoio.VideoCapture;
import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;


@RestController
public class HomePage {

    @GetMapping("/")
    public String getHomePage(){
        OpenCV.loadLocally();

        VideoCapture camera = new VideoCapture(0); // 0 = default webcam

        if (!camera.isOpened()) {
            System.out.println("Camera not found!");
            return "No camera";
        }

        Mat frame = new Mat();

        while (true) {
            camera.read(frame);

            if (!frame.empty()) {
                HighGui.imshow("Camera Feed", frame);
            }

            if (HighGui.waitKey(30) == 27) { // Press ESC to exit
                break;
            }
        }

        camera.release();
        HighGui.destroyAllWindows();
    }
    
}
