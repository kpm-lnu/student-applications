import React from "react";
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import { Paper } from "@mui/material";
import { SavedSimulation } from "../../../../shared/model/SavedSimulation";

//TODO. Implement saving results.
export const SavedSimulationTable = (): React.JSX.Element => {
  const mockedData: SavedSimulation[] = [new SavedSimulation(1, 'test')]


  return (
    <TableContainer component={Paper}>
      <TableHead>
      <TableRow>
        <TableCell>Simulation Id</TableCell>
        <TableCell>Simulation Title</TableCell>
      </TableRow>
      </TableHead>
      <TableBody>
        {mockedData.map((row) => (
          <TableRow
            key={row.id}
            sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
          >
            <TableCell component="th" scope="row">
              {row.id}
            </TableCell>
            <TableCell component="th" scope="row">
              {row.simulationName}
            </TableCell>

          </TableRow>
        ))}
      </TableBody>
    </TableContainer>
  )
}
