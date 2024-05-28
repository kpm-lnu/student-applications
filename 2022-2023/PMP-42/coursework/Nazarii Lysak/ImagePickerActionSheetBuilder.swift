//
//  ImagePickerActionSheetBuilder.swift
//  Emotion Recognition
//
//  Created by Nazar Lysak on 1/10/23.
//

import Foundation
import UIKit

// MARK: - ImagePickerActionSheetBuilderDelegate
protocol ImagePickerActionSheetBuilderDelegate: AnyObject {
    
    func imagePickerControllerDidCancel()
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any])
}

// MARK: - ImagePickerActionSheetBuilder
class ImagePickerActionSheetBuilder: NSObject {
    
    // MARK: - Types
    
    private struct Constants {
        
        static let cameraActionTitle = "Camera"
        static let photosActionTitle = "Photos"
        static let cancelActionTitle = "Cancel"
    }
    
    // MARK: - Properties
    
    private(set) var alertController: UIAlertController
    
    weak var delegate: ImagePickerActionSheetBuilderDelegate?
    
    // MARK: - Initialization
    
    init(title: String? = nil, message: String? = nil) {
        alertController = UIAlertController(title: title, message: message, preferredStyle: .actionSheet)
    }
    
    // MARK: - Public methods
    
    func addCameraAction(viewController: UIViewController?) -> ImagePickerActionSheetBuilder {
        
        let action = UIAlertAction(title: Constants.cameraActionTitle, style: .default, handler: { action in
            
            let vc = UIImagePickerController()
            
            vc.delegate = self
            vc.sourceType = UIImagePickerController.isSourceTypeAvailable(.camera) ? .camera : .photoLibrary
            
            viewController?.present(vc, animated: true)
        })
        
        alertController.addAction(action)
        return self
    }
    
    func addPhotosActionTitle(viewController: UIViewController?) -> ImagePickerActionSheetBuilder {
        
        let action = UIAlertAction(title: Constants.photosActionTitle, style: .default, handler: { action in
            let vc = UIImagePickerController()
            
            vc.delegate = self
            vc.sourceType = .photoLibrary
            
            viewController?.present(vc, animated: true)
        })
        
        alertController.addAction(action)
        return self
    }
    
    func addCancelAction(style: UIAlertAction.Style = .cancel, handler: (() -> ())? = nil) -> ImagePickerActionSheetBuilder {
        
        let action = UIAlertAction(title: Constants.cancelActionTitle, style: style, handler: { action in
            handler?()
        })
        
        alertController.addAction(action)
        return self
    }
    
    func present(from viewController: UIViewController?, animated: Bool = true) {
        viewController?.present(alertController, animated: animated)
    }
}

// MARK: - UIImagePickerControllerDelegate, UINavigationControllerDelegate
extension ImagePickerActionSheetBuilder: UIImagePickerControllerDelegate, UINavigationControllerDelegate {

    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        delegate?.imagePickerControllerDidCancel()
    }

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        delegate?.imagePickerController(picker, didFinishPickingMediaWithInfo: info)
    }
}
