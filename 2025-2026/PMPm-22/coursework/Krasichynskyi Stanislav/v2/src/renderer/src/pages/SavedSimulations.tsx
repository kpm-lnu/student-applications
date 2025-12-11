import { Box, Typography } from "@mui/material";
import { SavedSimulationTable } from "@renderer/components/menu/SavedSimulationTable";

export const SavedSimulations = (): React.JSX.Element => {
  return <Box
    sx={{
      width: '100%',
      height: '100%',
      p: 4
    }}
  >
  <Typography >Saved Simulations:</Typography>
    <SavedSimulationTable/>
  </Box>
}
