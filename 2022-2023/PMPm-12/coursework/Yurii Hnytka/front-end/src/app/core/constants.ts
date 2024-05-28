import * as cytoscape from 'cytoscape';
import * as klay from 'cytoscape-klay';

export const cytoscapeOptions: cytoscape.CytoscapeOptions = {
    minZoom: 0.1,
    maxZoom: 3,
    zoom: 1,
    wheelSensitivity: 0.2,
    style: [
        {
            selector: 'node',
            style: {
                label: 'data(id)',
                'font-weight': 'bold',
                color: 'white',
                width: '100px',
                height: '50px',
                backgroundColor: '#d60053',
                'background-blacken': 0.2,
                shape: 'round-rectangle',
            },
        },
        {
            selector: 'node:selected',
            style: {
                backgroundColor: 'blue',
                'background-blacken': 0,
            },
        },
        {
            selector: 'edge',
            style: {
                width: 3,
                'curve-style': 'bezier',
                'target-arrow-shape': 'triangle-backcurve',
                'arrow-scale': 2,
                color: 'white',
            },
        },
    ],
    boxSelectionEnabled: false,
    styleEnabled: true,
};

export const cytoscapeLayoutOptions: klay.KlayLayoutOptions = {
    name: 'klay',
    fit: true,
    animate: true,
    animationEasing: 'ease-in-out',
    animationDuration: 300,
    klay: {
        spacing: 100,
    },
};
