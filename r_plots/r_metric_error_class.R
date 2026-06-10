library(dplyr)
library(ggplot2)

# Load the data exported from Python
sumEvalResAllDf_filtered <- read.csv('data/rule_eval/output/step2AndStep3Data.csv')

# Add errRuleStr column if it doesn't exist
if (!"errRuleStr" %in% names(sumEvalResAllDf_filtered)) {
  # Create errRule string mapping (abbreviated codes)
  errRule_str_map <- c(
    ";explicitMissingValue" = "emv",
    ";disguisedMissingValues" = "dmv",
    ";contradictions" = "con",
    ";Misfielded_Value" = "mfv",
    ";embeddedValue" = "ebv",
    ";spellingMistake" = "spm",
    ";Domain_Violation" = "dov",
    ";Incorrect_Format" = "ifo",
    ";Incorrect_Encoding" = "ics"
  )
  sumEvalResAllDf_filtered$errRuleStr <- errRule_str_map[sumEvalResAllDf_filtered$errRule_clean]
  # Handle any NA (values not in the mapping) by keeping original
  sumEvalResAllDf_filtered$errRuleStr[is.na(sumEvalResAllDf_filtered$errRuleStr)] <- 
    sumEvalResAllDf_filtered$errRule_clean[is.na(sumEvalResAllDf_filtered$errRuleStr)]
}

# Melt all metrics including F1 if needed
value_vars <- c("mean_f1_per_errRule", "mean_f1_per_errRow", "mean_recall_per_errRule")

id_vars_li <- c("errRule", "fileInfo_cat", "errRule_clean", "errRuleStr", "Modelname_short", "datasetName")

plot_data <- sumEvalResAllDf_filtered %>%
  tidyr::pivot_longer(
    cols = all_of(value_vars),
    names_to = "name",
    values_to = "value"
  )

# Rename the 'name' column values
name_map <- c(
  "mean_precision_per_errRule" = "Precision per Error Class",
  "mean_recall_per_errRule" = "Recall per Error Class",
  "mean_precision_per_errRow" = "Precision per Row",
  "mean_recall_per_errRow" = "Recall per Row",
  "mean_f1_per_errRule" = "F1 per Error Class",
  "mean_f1_per_errRow" = "F1 per Row"
)

plot_data$name <- name_map[plot_data$name]
plot_data$name[is.na(plot_data$name)] <- plot_data$name[is.na(plot_data$name)]

# Filter for F1 per Row
step2_data <- plot_data %>% filter(name == 'F1 per Row')
yStr <- 'F1'

# Filter out PCI for comparison
step2_data_filtered <- step2_data

# Capitalize first letter of datasetName
step2_data_filtered$datasetName <- paste0(toupper(substr(step2_data_filtered$datasetName, 1, 1)), substr(step2_data_filtered$datasetName, 2, nchar(step2_data_filtered$datasetName)))

# Plot: F1 per Row
g2  <- ggplot(step2_data_filtered, aes(x = errRuleStr, y = value, shape = Modelname_short, color = Modelname_short)) +
  geom_point(size = 3, stroke = 2) +
  xlab('Error Class') + ylab(yStr) + ylim(0, 1) +
  facet_grid(rows = vars(fileInfo_cat), cols = vars(datasetName)) +
  theme_bw() +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 14),
    axis.text.y = element_text(size = 14),
    axis.title = element_text(size = 16),
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 16),
    strip.text = element_text(size = 14)
  ) +
  scale_color_brewer(palette='Set2') +
  scale_shape_manual(values = c(0, 2, 4)) +
  labs(shape = 'Metric', color = 'Metric') +
  guides(shape = guide_legend(nrow = 1), color = guide_legend(nrow = 1))

# Save the plot
output_dir <- 'data/rule_eval/output/'
ggsave(g2, filename = paste0(output_dir, 'step2_paramNoparam.jpg'), width=12, height = 6, dpi = 300)

# Plot: Recall per Error Class
step3_data <- plot_data %>% filter(name == 'Recall per Error Class')
step3_data$datasetName <- paste0(toupper(substr(step3_data$datasetName, 1, 1)), substr(step3_data$datasetName, 2, nchar(step3_data$datasetName)))

g3 <- ggplot(step3_data, aes(x = errRuleStr, y = value, color = Modelname_short, shape = Modelname_short)) +
  geom_point(size = 3, stroke = 2) +
  xlab('Error Class') + ylab('Recall') + ylim(0, 1) +
  facet_grid(rows = vars(fileInfo_cat), cols = vars(datasetName)) +
  theme_bw() +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 14),
    axis.text.y = element_text(size = 14),
    axis.title = element_text(size = 16),
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 16),
    strip.text = element_text(size = 14)
  ) +
  scale_color_brewer(palette='Set2') +
  scale_shape_manual(values = c(0, 2, 4)) +
  labs(color = 'Metric', shape = 'Metric') +
  guides(shape = guide_legend(nrow = 1), color = guide_legend(nrow = 1))

# Save the plot
ggsave(g3, filename = paste0(output_dir, 'step3_paramNoparam.jpg'), width=12, height = 6, dpi = 300)