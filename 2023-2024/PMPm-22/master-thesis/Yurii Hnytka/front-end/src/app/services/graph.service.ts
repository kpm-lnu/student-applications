import { EventEmitter, Injectable } from '@angular/core';
import { ICompartment, IFlow } from '@core/types/model.types';
import cytoscape, {
    EdgeCollection,
    EdgeSingular,
    EventObject,
    NodeSingular,
    SingularElementArgument,
} from 'cytoscape';
import edgehandles from 'cytoscape-edgehandles';
import klay from 'cytoscape-klay';
import { cytoscapeLayoutOptions, cytoscapeOptions } from '../core/constants';

cytoscape.use(klay);
cytoscape.use(edgehandles);

@Injectable({
    providedIn: 'root',
})
export class GraphService {
    private cytoscapeObj!: cytoscape.Core;
    private edgehandles!: edgehandles.EdgeHandlesInstance;

    public readonly compartmentOpen: EventEmitter<NodeSingular> =
        new EventEmitter<NodeSingular>();
    public readonly flowOpen: EventEmitter<EdgeSingular> =
        new EventEmitter<EdgeSingular>();
    public readonly flowAdd: EventEmitter<EdgeSingular> =
        new EventEmitter<EdgeSingular>();

    public get graphReady(): boolean {
        return this.cytoscapeObj !== undefined;
    }
    public get selectedElement(): SingularElementArgument {
        return this.cytoscapeObj.elements(':selected').first();
    }

    public init(container: HTMLElement): void {
        this.cytoscapeObj = cytoscape({
            container: container,
            ...cytoscapeOptions,
        });

        this.edgehandles = this.cytoscapeObj.edgehandles();

        this.cytoscapeObj.on('dblclick', this.onDBClick.bind(this));
        this.cytoscapeObj.on('cxttapend', 'node', this.onFlowStart.bind(this));
        this.cytoscapeObj.on('ehcomplete', this.onFlowEnd.bind(this) as any);
        this.cytoscapeObj.on('select', 'node, edge', this.onSelect.bind(this));
    }

    public layout(): void {
        this.cytoscapeObj.layout(cytoscapeLayoutOptions).run();
    }

    public clear(): void {
        this.cytoscapeObj.elements().remove();
    }

    public addCompartment(
        compartment: ICompartment,
        layout: boolean = true
    ): void {
        this.cytoscapeObj.add({
            group: 'nodes',
            data: {
                id: compartment.id,
                name: compartment.name,
                value: compartment.value,
            },
            classes: 'compartment',
            renderedPosition: {
                x: this.cytoscapeObj.width() / 2,
                y: this.cytoscapeObj.height() / 2,
            },
        });

        if (layout) {
            this.layout();
        }
    }

    public updateCompartment(compartment: ICompartment): void {
        const node: NodeSingular = this.cytoscapeObj
            .nodes()
            .$id(compartment.id);

        this.renameDefinitionInFlows(node.data('name'), compartment.name);

        node.data({
            name: compartment.name,
            value: compartment.value,
        });
    }

    public removeCompartment(id: string, layout: boolean = true): void {
        const node: NodeSingular = this.cytoscapeObj.nodes().$id(id);

        this.removeFlowsByDefinition(node.data('name'));

        node.remove();

        if (layout) {
            this.layout();
        }
    }

    public addFlow(flow: IFlow, layout: boolean = true): void {
        this.cytoscapeObj.add({
            group: 'edges',
            data: {
                id: flow.id,
                source: flow.source,
                target: flow.target,
                equation: flow.equation,
            },
        });

        if (layout) {
            this.layout();
        }
    }

    public updateFlow(flow: IFlow, layout: boolean = true): void {
        const edge: EdgeSingular = this.cytoscapeObj.edges().$id(flow.id);

        edge.data('equation', flow.equation);

        edge.move({
            source: flow.source,
            target: flow.target,
        });

        if (layout) {
            this.layout();
        }
    }

    public removeFlow(id: string, layout: boolean = false): void {
        this.cytoscapeObj.edges().$id(id).remove();

        if (layout) {
            this.layout();
        }
    }

    private removeFlowsByDefinition(name: string): void {
        const expressionRegExp: RegExp = new RegExp(`\\b${name}\\b`, 'g');
        const edges: EdgeCollection = this.cytoscapeObj
            .edges()
            .filter((edge: EdgeSingular): boolean =>
                edge.data('equation').match(expressionRegExp)
            );

        edges.remove();
    }

    private renameDefinitionInFlows(name: string, newName: string): void {
        const expressionRegExp: RegExp = new RegExp(`\\b${name}\\b`, 'g');

        this.cytoscapeObj.edges().forEach((edge: EdgeSingular): void => {
            edge.data(
                'equation',
                edge.data('equation').replace(expressionRegExp, newName)
            );
        });
    }

    private onDBClick(event: EventObject): void {
        if (!event.target.group) {
            return;
        }

        if (event.target.group() === 'nodes') {
            this.compartmentOpen.next(event.target as NodeSingular);

            return;
        }

        this.flowOpen.next(event.target as EdgeSingular);
    }

    private onFlowStart(event: EventObject): void {
        this.edgehandles.start(event.target);
    }

    private onFlowEnd(
        event: EventObject,
        sourceNode: NodeSingular,
        targetNode: NodeSingular,
        addedEdge: EdgeSingular
    ): void {
        this.flowAdd.next(addedEdge);
    }

    private onSelect(event: EventObject): void {
        this.cytoscapeObj.elements().not(event.target).unselect();
    }
}
