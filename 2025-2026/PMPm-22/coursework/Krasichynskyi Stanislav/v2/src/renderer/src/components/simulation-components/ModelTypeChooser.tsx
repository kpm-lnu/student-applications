import { Box, Button, Typography } from '@mui/material'
import { useState } from 'react'
import { ModelType } from '../../../../shared/enums'

export const ModelTypeChooser = (): React.JSX.Element => {
  const [selectedModelType, setSelectedModelType] = useState(ModelType.None)

  const handleButtonClick = (typeIndex: number) => {
    let newType: ModelType;

    switch (typeIndex) {
      case 1:
        newType = ModelType.OneDimensional;
        break;
      case 2:
        newType = ModelType.TwoDimensional;
        break;
      case 3:
        newType = ModelType.ThreeDimensional;
        break;
      default:
        newType = ModelType.None;
        break;
    }

    setSelectedModelType(newType);
    console.log('ModelType was set to:', newType);

    //TODO. Replace
    // SimulationService.getInstance().setModelType(newType);
  }

  return (
    <>
      <Typography variant="h5">Choose type of model</Typography>
      <Box sx={{ display: 'flex', gap: 2 }}>
        <Button onClick={() => handleButtonClick(1)} disabled={selectedModelType === ModelType.OneDimensional} variant="contained">
          One Dimensional
        </Button>
        <Button onClick={() => handleButtonClick(2)} disabled={selectedModelType === ModelType.TwoDimensional} variant="contained">
          Two Dimensional
        </Button>
        <Button onClick={() => handleButtonClick(3)} disabled={selectedModelType === ModelType.ThreeDimensional} variant="contained">
          Three Dimensional
        </Button>
      </Box>
    </>
  )
}
