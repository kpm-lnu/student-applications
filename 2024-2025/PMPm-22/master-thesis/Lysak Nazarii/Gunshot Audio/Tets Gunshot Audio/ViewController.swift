//
//  ViewController.swift
//  Tets Gunshot Audio
//
//  Created by Nazar Lysak on 02.01.2025.
//

import UIKit
import AVKit
import CoreML
import AVFoundation
import Vision
import SoundAnalysis

// MARK: - ViewController
class ViewController: UIViewController {
    
    // MARK: - Outlets
    
    @IBOutlet private weak var titleLabel: UILabel!
    @IBOutlet private weak var resultLabel: UILabel!
    
    // MARK: - Properties
    
    private let audioEngine = AVAudioEngine()
    private var soundClassifier = try? GunshotSoundClassifier()

    private var inputFormat: AVAudioFormat!
    private var analyzer: SNAudioStreamAnalyzer!
    private var resultsObserver = ResultsObserver()
    private let analysisQueue = DispatchQueue(label: "com.custom.AnalysisQueue")
    
    // MARK: - Lifecycle
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        resultsObserver.delegate = self
        inputFormat = audioEngine.inputNode.inputFormat(forBus: 0)
        analyzer = SNAudioStreamAnalyzer(format: inputFormat)
    }
    
    override func viewDidAppear(_ animated: Bool) {
        startAudioEngine()
    }
}

// MARK: - Private methods
private extension ViewController {
    
    func startAudioEngine() {
        
        guard
            let soundClassifier = soundClassifier
        else {
            return
        }
        
        do {
            let request = try SNClassifySoundRequest(mlModel: soundClassifier.model)
            try analyzer.add(request, withObserver: resultsObserver)
            
        } catch {
            
            print("Unable to prepare request: \(error.localizedDescription)")
            return
        }
        
        audioEngine.inputNode.installTap(onBus: 0, bufferSize: 8000, format: inputFormat) { buffer, time in
            
            guard let channelData = buffer.floatChannelData?[0] else { return }
            let channelDataValue = UnsafeBufferPointer(start: channelData, count: Int(buffer.frameLength))
            
            // Обчислення середньоквадратичного значення
            var sum: Float = 0.0
            for sample in channelDataValue {
                sum += sample * sample
            }
            
            let rms = sqrt(sum / Float(buffer.frameLength))
            let avgPower = 20 * log10(rms)
            
            DispatchQueue.main.async {
                if avgPower > -40 {
                    self.analysisQueue.async {
                        self.analyzer.analyze(buffer, atAudioFramePosition: time.sampleTime)
                    }
                } else {
                    self.resultLabel.text = ("Recognition: Urban \nConfidence 100")
                }
            }
            
        }
        
        do {
            try audioEngine.start()
            
        } catch(_) {
            print("error in starting the Audio Engin")
        }
    }
}

// MARK: - GenderClassifierDelegate
extension ViewController: GenderClassifierDelegate {
    
    func displayPredictionResult(identifier: String, confidence: Double) {
        
        DispatchQueue.main.async {
            self.resultLabel.text = ("Recognition: \(identifier)\nConfidence \(confidence)")
        }
    }
}
