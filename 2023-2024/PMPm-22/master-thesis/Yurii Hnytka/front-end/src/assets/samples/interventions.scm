{
    "compartments": [
        {
            "id": "ba3aa850-1bdc-571c-b3e2-ae3f99e08677",
            "name": "S",
            "value": 35200000
        },
        {
            "id": "ffb8d7d5-4f36-5826-953b-7a655749784d",
            "name": "E1",
            "value": 0
        },
        {
            "id": "5bdb27da-a3fc-57ee-9638-ca7910ca46f4",
            "name": "E2",
            "value": 0
        },
        {
            "id": "d1b2c7bb-7628-50cb-aa37-6097303255f9",
            "name": "I1",
            "value": 3520000
        },
        {
            "id": "80fa5481-1a4d-5550-a955-4a5f8845e1bb",
            "name": "I2",
            "value": 5280000
        },
        {
            "id": "aff45979-5966-5a9b-b99d-fdbd8ad2a0fe",
            "name": "Casual",
            "value": 0
        },
        {
            "id": "d2cd90f8-b773-529a-a9b1-512c165c53bc",
            "name": "Fatal",
            "value": 0
        },
        {
            "id": "0dac27dc-02a0-5f56-a05c-a1a1de406335",
            "name": "Hosp",
            "value": 0
        },
        {
            "id": "82801f83-4814-534f-a8df-22554dcbc576",
            "name": "R1",
            "value": 0
        },
        {
            "id": "bf8bf2c7-7c97-5ea5-9887-f1f4f8cf4c7d",
            "name": "R2",
            "value": 0
        },
        {
            "id": "91f397f4-eee8-5ce3-906e-e65cf30bc944",
            "name": "Death",
            "value": 0
        },
        {
            "id": "91bae6a1-1aea-55af-8ebf-eb39cf0a0b85",
            "name": "Immunity",
            "value": 0
        }
    ],
    "interventions": [
        {
            "id": "ca05e80e-32c3-5598-8f04-486818880451",
            "name": "u1"
        },
        {
            "id": "c97d380e-ea16-5744-8be6-37d61c9da852",
            "name": "u2"
        }
    ],
    "constants": [
        {
            "id": "9acb26f0-3766-5d31-b582-ae47d02f1c3d",
            "name": "c",
            "value": 1000
        },
        {
            "id": "f4e4d128-f17b-5263-84b5-f6b90c3cf632",
            "name": "d1",
            "value": 300
        },
        {
            "id": "9692ef56-0bae-5f77-adeb-be01f4dfa6b2",
            "name": "d2",
            "value": 1500
        },
        {
            "id": "38db1106-3d24-5a30-8dd7-cb9b4e78d485",
            "name": "alpha",
            "value": 0.7
        },
        {
            "id": "c0663c5f-7379-5ca8-ab23-318509cdf56a",
            "name": "beta",
            "value": 0.3
        },
        {
            "id": "245cff4c-fde2-5364-aa0d-db33e4b96410",
            "name": "Tvaccine_lag",
            "value": 28
        },
        {
            "id": "9a8a85d0-673c-58d8-b109-faf1af790957",
            "name": "R0",
            "value": 2.25
        },
        {
            "id": "020e08d8-ecdc-5be7-8ec4-b1299e412d0a",
            "name": "N",
            "value": 44e6
        },
        {
            "id": "80b2bf75-d235-525a-a5c1-8d8471cf3be6",
            "name": "Tinf",
            "value": 10
        },
        {
            "id": "7b742e7d-f2f2-5d28-b33e-da7114a6b70a",
            "name": "Tinc",
            "value": 10
        },
        {
            "id": "01e6f71b-0b5f-5aef-9b05-bbad34e2561a",
            "name": "Tquar",
            "value": 10
        },
        {
            "id": "ff3eca0d-cde3-53ac-aeaf-ca29c888e909",
            "name": "Trecovery_casual",
            "value": 11
        },
        {
            "id": "452c7628-fa81-5326-9ddc-b8b78a2d8451",
            "name": "Thospital_lag",
            "value": 30
        },
        {
            "id": "259d949f-4f26-5217-9e53-630a87ff1d56",
            "name": "Tdeath",
            "value": 5
        },
        {
            "id": "1de949eb-dd29-5a0a-ad54-d0cac1519a6a",
            "name": "pAsym",
            "value": 0.4
        },
        {
            "id": "e439ed9f-e027-5d3c-b097-ae4b8f45a719",
            "name": "pCasual",
            "value": 0.8
        },
        {
            "id": "29f1be42-965f-5a4f-9631-590bfcfe8d4e",
            "name": "pFatal",
            "value": 0.2
        },
        {
            "id": "746cd545-c87b-597f-8e29-aa25031432ee",
            "name": "Trenewal",
            "value": 5
        }
    ],
    "flows": [
        {
            "id": "f4fa7f56-79f6-567e-9fb6-670e6595539c",
            "equation": "pAsym * (1 - u1 + (1 - alpha)*u1)^2 * (R0 * S*(I1 + I2))/(Tinf * N)",
            "source": "ba3aa850-1bdc-571c-b3e2-ae3f99e08677",
            "target": "ffb8d7d5-4f36-5826-953b-7a655749784d"
        },
        {
            "id": "0270abd0-fa0e-534b-b705-5cc583c6816b",
            "equation": "(1 - pAsym) * (1 - u1 + (1 - alpha)*u1)^2 * (R0 * S*(I1 + I2))/(Tinf * N)",
            "source": "ba3aa850-1bdc-571c-b3e2-ae3f99e08677",
            "target": "5bdb27da-a3fc-57ee-9638-ca7910ca46f4"
        },
        {
            "id": "13064f46-7dda-599f-a16d-766c608fc97b",
            "equation": "u2 * S * beta / Tvaccine_lag",
            "source": "ba3aa850-1bdc-571c-b3e2-ae3f99e08677",
            "target": "91bae6a1-1aea-55af-8ebf-eb39cf0a0b85"
        },
        {
            "id": "4915b80c-d244-5454-b0dc-faf0166023cb",
            "equation": "E1 / Tinc",
            "source": "ffb8d7d5-4f36-5826-953b-7a655749784d",
            "target": "d1b2c7bb-7628-50cb-aa37-6097303255f9"
        },
        {
            "id": "0870d136-9b45-59d7-a748-8bcce9257cdc",
            "equation": "E2 / Tinc",
            "source": "5bdb27da-a3fc-57ee-9638-ca7910ca46f4",
            "target": "80fa5481-1a4d-5550-a955-4a5f8845e1bb"
        },
        {
            "id": "a8d70c82-0db7-5d34-8052-d1a0a060ca85",
            "equation": "I1 / Tinf",
            "source": "d1b2c7bb-7628-50cb-aa37-6097303255f9",
            "target": "82801f83-4814-534f-a8df-22554dcbc576"
        },
        {
            "id": "8ead0520-a67b-55b6-bc80-80a7a5310c2b",
            "equation": "R1 / Trenewal",
            "source": "82801f83-4814-534f-a8df-22554dcbc576",
            "target": "ba3aa850-1bdc-571c-b3e2-ae3f99e08677"
        },
        {
            "id": "212c6b47-2c27-545c-a1c9-f3bf9edbcfcd",
            "equation": "(pCasual)/(1 - pAsym) * I2 / Tquar",
            "source": "80fa5481-1a4d-5550-a955-4a5f8845e1bb",
            "target": "aff45979-5966-5a9b-b99d-fdbd8ad2a0fe"
        },
        {
            "id": "d5224b0f-0cbe-5025-9626-1d86a2acfc58",
            "equation": "(pFatal)/(1 - pAsym) * I2 / Tquar",
            "source": "80fa5481-1a4d-5550-a955-4a5f8845e1bb",
            "target": "d2cd90f8-b773-529a-a9b1-512c165c53bc"
        },
        {
            "id": "671ae7f4-d322-5a7c-a3c6-33c4bcd0e5d5",
            "equation": "Casual / Trecovery_casual",
            "source": "aff45979-5966-5a9b-b99d-fdbd8ad2a0fe",
            "target": "bf8bf2c7-7c97-5ea5-9887-f1f4f8cf4c7d"
        },
        {
            "id": "85867b95-d2de-5b39-af18-62419438dc6b",
            "equation": "R2 / Trenewal",
            "source": "bf8bf2c7-7c97-5ea5-9887-f1f4f8cf4c7d",
            "target": "ba3aa850-1bdc-571c-b3e2-ae3f99e08677"
        },
        {
            "id": "53954385-be21-524a-be38-9f00aeaf9309",
            "equation": "Fatal / Thospital_lag",
            "source": "d2cd90f8-b773-529a-a9b1-512c165c53bc",
            "target": "0dac27dc-02a0-5f56-a05c-a1a1de406335"
        },
        {
            "id": "f261ce74-9835-56f3-ba8f-00a327915910",
            "equation": "Hosp / Tdeath",
            "source": "0dac27dc-02a0-5f56-a05c-a1a1de406335",
            "target": "91f397f4-eee8-5ce3-906e-e65cf30bc944"
        }
    ]
}
