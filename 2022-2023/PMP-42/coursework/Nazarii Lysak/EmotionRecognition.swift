//
//  EmotionRecognition.swift
//  Emotion Recognition
//
//  Created by Nazar Lysak on 4/13/23.
//

import Foundation
import UIKit
import ARKit

class EmotionRecognition {
    
    // MARK: - Types
    
    struct Constants {
        
        static let cannotRecognizeEmotionText = "Can't recognize"
    }
    
    // MARK: - Properties
    
    private var model: VNCoreMLModel?
    
    // MARK: - Initialization
    
    init() {
        
        do {
            model = try VNCoreMLModel(for: CNNEmotions().model)
        } catch {
            print("Can't load Vision ML model: \(error)")
        }
    }
    
    // MARK: - Public methods
    
    func recognizeEmotion(at image: UIImage?, completion: @escaping (String) -> ()) {
        
        guard
            let model = model,
            let image = image,
            let pixelBuffer = image.pixelBuffer()
        else {
            
            completion(Constants.cannotRecognizeEmotionText)
            return
        }
        
        let options = [CIDetectorAccuracy: CIDetectorAccuracyHigh]
        let faceDetector = CIDetector(ofType: CIDetectorTypeFace, context: nil, options: options)!

        let ciImage = CIImage(cgImage: image.cgImage!)
        let faces = faceDetector.features(in: ciImage)

//        guard
//            let face = faces.first as? CIFaceFeature
//        else {
//
//            completion(Constants.cannotRecognizeEmotionText)
//            return
//        }
                
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:]).perform([VNCoreMLRequest(model: model) { request, error in
            
            guard
                let firstResult = (request.results as? [VNClassificationObservation])?.first
            else {
                
                completion(Constants.cannotRecognizeEmotionText)
                return
            }

            DispatchQueue.main.async {

                let emotionText = firstResult.confidence > 0.70 ? firstResult.identifier : Constants.cannotRecognizeEmotionText
                completion(emotionText)
            }
        }])
    }
    
}
