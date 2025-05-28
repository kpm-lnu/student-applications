//
//  ResultsObserver.swift
//  Tets Gunshot Audio
//
//  Created by Nazar Lysak on 02.01.2025.
//

import Foundation
import SoundAnalysis

// MARK: - GenderClassifierDelegate
protocol GenderClassifierDelegate {
    func displayPredictionResult(identifier: String, confidence: Double)
}

// MARK: - ResultsObserver
class ResultsObserver: NSObject, SNResultsObserving {
    
    var delegate: GenderClassifierDelegate?
    
    func request(_ request: SNRequest, didProduce result: SNResult) {
        
        guard
            let result = result as? SNClassificationResult,
            let classification = result.classifications.first
        else {
            return
        }
        
        let confidence = classification.confidence * 100.0
        
        if confidence > 60 {
            
            print(result.classifications)
            delegate?.displayPredictionResult(identifier: classification.identifier, confidence: confidence)
        }
    }
}
