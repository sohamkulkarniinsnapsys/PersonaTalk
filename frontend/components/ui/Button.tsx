import React from 'react';

export type ButtonVariant = 'primary' | 'secondary' | 'danger';

export type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
};

export function Button({ variant = 'secondary', className = '', ...props }: ButtonProps) {
  const variantClass =
    variant === 'primary'
      ? 'ui-button-primary'
      : variant === 'danger'
        ? 'ui-button-danger'
        : 'ui-button-secondary';

  return <button {...props} className={`${variantClass} ${className}`.trim()} />;
}
