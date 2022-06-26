apiVersion: argoproj.io/v1alpha1
kind: Sensor
metadata:
  name: kafka-front
  namespace: argo-demo
spec:
  dependencies:
    - name: test-dep
      eventSourceName: kafka-front
      eventName: kafka-es-front
  triggers:
    - template:
        name: argo-workflow-trigger
        argoWorkflow:
          source:
            resource:
              apiVersion: argoproj.io/v1alpha1
              kind: Workflow
              metadata:
                name: test
                namespace: argo-demo
              spec:
                entrypoint: test
                securityContext:
                  runAsNonRoot: true
                  runAsUser: 8737
                imagePullSecrets:
                - name: gitlab-regcred
                serviceAccountName: argo
                templates:
                - name: test
                  dag:
                    tasks:
                    - name: auto-seg
                  template: model1
                    - name: output
                  dependencies: [auto-seg]
                  template: sender
                - name: model1
                  script:
                    args:
                  - >-
                    curl -X POST
                    'http://10.130.0.39:5001/process'
                    command:
                  - sh
                  - '-c'
                    image: >-
                  registry.gitlab.com/neuristix/prototyping/pipelines/argo-workflows/curl_for_webhook:v1
                    imagePullPolicy: Always
                - name: sender
                  script:
                    args:
                  - >-
                    curl -d "{\"file\":\"$MESSAGE\",
                    \"callback\":\"$CALLBACK\"}" -H
                    'Content-Type:application/json' -X POST
                    'http://10.130.0.39:12000/result'
                    command:
                  - sh
                  - '-c'
                    env:
                  - name: MESSAGE
                    value: >-
                      http://gpu-v100.dev.pbd.ai:9001/data/results/browse
                  - name: CALLBACK
                    value: callback-default
                    image: >-
                  registry.gitlab.com/neuristix/prototyping/pipelines/argo-workflows/curl_for_webhook:v1
                    imagePullPolicy: Always
          operation: submit
          parameters:
            - src:
                dependencyName: test-dep
                dataKey: body.short_file
              dest: spec.templates.0.dag.tasks.0.arguments.artifacts.0.s3.key
            - src:
                dependencyName: test-dep
                dataKey: body.short_file
              dest: spec.templates.0.dag.tasks.1.arguments.artifacts.0.s3.key
            - src:
                dependencyName: test-dep
                dataKey: body.short_file
              dest: spec.templates.0.dag.tasks.2.arguments.artifacts.0.s3.key
            - src:
                dependencyName: test-dep
                dataKey: body.callback
              dest: spec.templates.10.script.env.1.value
