{{/*
Expand the name of the chart.
*/}}
{{- define "continuous-image-gen.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "continuous-image-gen.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "continuous-image-gen.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "continuous-image-gen.labels" -}}
helm.sh/chart: {{ include "continuous-image-gen.chart" . }}
{{ include "continuous-image-gen.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
cso.platform/module: continuous-image-gen
cso.platform/tier: {{ .Values.csoModule.tier | default "custom" }}
cso.platform/category: {{ .Values.csoModule.category | default "ai-ml" }}
cso.platform/environment: {{ .Values.global.cso.environment | default "development" }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "continuous-image-gen.selectorLabels" -}}
app.kubernetes.io/name: {{ include "continuous-image-gen.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "continuous-image-gen.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "continuous-image-gen.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}