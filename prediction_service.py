class PredictionService:
  def predict(self, image_path):
      # Returns the following for each frame(image):
      # is_box T/F (don't process further if it isn't box)
      # prediction_confidence_score (prediction_confidence_score in percentage)
      # status (return either 'finished' or 'errored')
      #above_pallet_thresh (T/F)
      pass
