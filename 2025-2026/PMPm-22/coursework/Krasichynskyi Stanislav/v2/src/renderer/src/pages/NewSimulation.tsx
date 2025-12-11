import { Box, Button, Step, StepLabel, Stepper, Typography } from "@mui/material";
import { useState } from "react";
import { ModelTypeChooser } from "@renderer/components/simulation-components/ModelTypeChooser";

export const NewSimulation = (): React.JSX.Element => {
  const [activeStep, setActiveStep] = useState(0);

  const steps = [
    'Model Type',
    'Area',
    'Mesh',
    'Select Math Model',
    'Configure Model',
    'Simulations',
  ]

  const handleNext = () => {
    if (activeStep < steps.length - 1) {
      setActiveStep(prev => prev + 1);
    }
  };
  const handleBack = () => {
    if (activeStep > 0) {
      setActiveStep(prev => prev - 1);
    }
  };

  return (
    <Box sx={{ width: '100%', p: 4 }}>
      <Stepper activeStep={activeStep} alternativeLabel>
        {steps.map(label => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      <Box
        sx={{
          mt: 4,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 2,
        }}
      >
        {activeStep === 0 && (
          <ModelTypeChooser/>
        )}

        {activeStep === 1 && (
          <Typography>Настройка області (Area)</Typography>
        )}
        {activeStep === 2 && (
          <Typography>Параметри сітки (Mesh)</Typography>
        )}
        {activeStep === 3 && (
          <Typography>Вибір математичної моделі</Typography>
        )}
        {activeStep === 4 && (
          <Typography>Конфігурація математичної моделі</Typography>
        )}
        {activeStep === 5 && (
          <Typography>Результати</Typography>
        )}
      </Box>

      <Box
        sx={{
          mt: 4,
          display: 'flex',
          justifyContent: 'space-between',
        }}
      >
        <Button
          disabled={activeStep === 0}
          onClick={handleBack}
        >
          Back
        </Button>
        <Button
          variant="contained"
          onClick={handleNext}
        >
          {activeStep === steps.length - 1 ? 'Finish' : 'Next'}
        </Button>
      </Box>
    </Box>
  );
}
