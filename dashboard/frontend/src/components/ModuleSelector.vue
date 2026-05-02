<script setup>
defineProps({
  modelValue: String,
})
const emit = defineEmits(['update:modelValue'])

const modules = [
  { value: 'encoder_intuition', label: 'Encoder + Intuition' },
  { value: 'reward_head', label: 'Reward Head' },
  { value: 'action_planner', label: 'Action Planner' },
  { value: 'metacontroller', label: 'Metacontroller' },
]
</script>

<template>
  <select :value="modelValue" @change="emit('update:modelValue', $event.target.value)">
    <option v-for="m in modules" :key="m.value" :value="m.value">{{ m.label }}</option>
  </select>
</template>
