library(dplyr)
library(ggplot2)
library(ggpattern)

# Load the data exported from Python
plot_df <- read.csv('data/rule_eval/output/step1_plot_df.csv')

# Create the plot with pattern fills for publication-quality appearance
ggplot(plot_df, 
       aes(x = datasetName, 
           y = Count_round, 
           fill = Rule_Type, 
           alpha = Modelname_short, 
           pattern = Rule_Type,
           group = interaction(Modelname_short, Rule_Type))) + 
  geom_col_pattern(
    width = 0.9, 
    position = position_dodge(width = 0.9), 
    # Adjust these two values to make the points look like a fine texture
    pattern_density = 0.1,
    pattern_spacing = 0.02, 
    pattern_size = 0.3 
  ) +
  scale_fill_manual(values = c(
    "Executable %" = "#f1a340",
    "Not Executable %" = "#998ec3"
  )) +
  # THIS IS THE KEY PART:
  scale_pattern_manual(values = c(
    "Executable %" = "circle", 
    "Not Executable %" = "none",
    "GLM-4.7" = "none",
    "Qwen3-Coder" = "none",
    "Gemma-4" = "none"
  )) +
  scale_alpha_manual(values = c(
    "GLM-4.7" = 0.6,
    "Qwen3-Coder" = 0.3,
    "Gemma-4" = 1
  )) +
  labs(
    title = "Rule Execution Status by Dataset",
    x = "Dataset",
    y = "Number of Rules",
    fill = "Rule Type",
    pattern = "Rule Type",
    alpha = "Model"
  ) +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 0, hjust = 0.5, size = 12),
    axis.text.y = element_text(size = 12),
    axis.title = element_text(size = 14),
    plot.title = element_text(size = 16),
    legend.position = "bottom",
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 12),
    strip.text = element_text(size = 12)
  ) +
  facet_grid(ruleInfoText ~ promptId) +
  guides(alpha = guide_legend(override.aes = list(pattern = "none")))

# Save the plot
output_dir <- 'data/rule_eval/output/'
ggsave(filename = paste0(output_dir,'step1_syntacticFilter.png'), width=12, height = 6, dpi = 300)