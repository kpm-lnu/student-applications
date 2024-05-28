//
//  ViewController.swift
//  Emotion Recognition
//
//  Created by Nazar Lysak on 1/10/23.
//

import UIKit

// MARK: - ViewController
class ViewController: UIViewController {
    
    // MARK: - Outlets
    
    @IBOutlet private weak var emotionLabel: UILabel!
    @IBOutlet private weak var userImageView: UIImageView!
}

// MARK: - Actions
private extension ViewController {
    
    @IBAction func onLoadImageButton(_ sender: Any) {
        
        let builder = ImagePickerActionSheetBuilder()
            .addCameraAction(viewController: self)
            .addPhotosActionTitle(viewController: self)
            .addCancelAction()
        
        builder.delegate = self
        builder.present(from: self)
    }
}

// MARK: - ImagePickerActionSheetBuilderDelegate
extension ViewController: ImagePickerActionSheetBuilderDelegate {
    
    func imagePickerControllerDidCancel() {
        dismiss(animated: true)
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        
        if let pickedImage = info[UIImagePickerController.InfoKey.originalImage] as? UIImage {
            userImageView.image = pickedImage
            
            EmotionRecognition().recognizeEmotion(at: pickedImage) { emotionText in
                self.emotionLabel.text = emotionText
            }
        }
        
        picker.dismiss(animated: true)
    }
}
